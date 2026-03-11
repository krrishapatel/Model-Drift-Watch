import os
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session

from .config import settings
from .models import Batch, Feature, QualityMetric, DriftMetric, ModelMetric, Alert, ReferenceStat, RollingStat
from .metrics import infer_schema
from .drift import build_reference_stats, compute_drift
from .quality import compute_quality
from .model_health import train_model, evaluate_model


def save_raw_file(file_bytes: bytes, batch_id: int):
    os.makedirs(settings.data_dir, exist_ok=True)
    path = os.path.join(settings.data_dir, f"batch_{batch_id}.csv")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def ensure_features(session: Session, schema: dict):
    features = {}
    for col in schema["columns"]:
        feature = session.query(Feature).filter(Feature.name == col["name"]).first()
        if not feature:
            feature = Feature(name=col["name"], type=col["type"])
            session.add(feature)
            session.flush()
        features[col["name"]] = feature
    return features


def get_reference_stats(session: Session, ref_batch_id: int | None):
    if not ref_batch_id:
        return None
    stats = {}
    rows = session.query(ReferenceStat).filter(ReferenceStat.batch_id == ref_batch_id).all()
    for row in rows:
        feature = session.query(Feature).filter(Feature.id == row.feature_id).first()
        if feature:
            stats[feature.name] = row.stats_json
    return stats


def get_rolling_stats(session: Session, current_batch_id: int):
    last_batch = session.query(Batch).filter(Batch.id < current_batch_id).order_by(Batch.id.desc()).first()
    if not last_batch:
        return None
    stats = {}
    rows = session.query(RollingStat).filter(RollingStat.batch_id == last_batch.id).all()
    for row in rows:
        feature = session.query(Feature).filter(Feature.id == row.feature_id).first()
        if feature:
            stats[feature.name] = row.stats_json
    return stats


def store_reference_stats(session: Session, batch_id: int, stats: dict, features: dict):
    session.query(ReferenceStat).filter(ReferenceStat.batch_id == batch_id).delete()
    for name, payload in stats.items():
        session.add(ReferenceStat(batch_id=batch_id, feature_id=features[name].id, stats_json=payload))


def store_rolling_stats(session: Session, batch_id: int, stats: dict, features: dict):
    for name, payload in stats.items():
        session.add(RollingStat(batch_id=batch_id, feature_id=features[name].id, stats_json=payload))


def create_alerts(session: Session, batch_id: int, quality_metrics: list, drift_metrics: list):
    alerts = []
    for qm in quality_metrics:
        if qm["metric"] == "missing_rate" and qm["value"] > settings.missing_rate_threshold:
            alerts.append(Alert(batch_id=batch_id, type="quality", severity="medium", status="open", message=f"Missing rate high for {qm['feature']}: {qm['value']:.2f}"))
    for dm in drift_metrics:
        if dm["metric"] == "psi" and dm["value"] > settings.psi_threshold:
            alerts.append(Alert(batch_id=batch_id, type="drift", severity="high", status="open", message=f"PSI drift on {dm['feature']}: {dm['value']:.2f}"))
        if dm["metric"] == "ks_pvalue" and dm["value"] < settings.ks_pvalue_threshold:
            alerts.append(Alert(batch_id=batch_id, type="drift", severity="medium", status="open", message=f"KS p-value low on {dm['feature']}: {dm['value']:.3f}"))
    for alert in alerts:
        session.add(alert)
    return alerts


def send_email_alerts(alerts: list):
    if not settings.alert_email_enabled or not alerts:
        return
    if not settings.smtp_host or not settings.alert_email_to:
        return
    import smtplib
    from email.mime.text import MIMEText

    body = "\n".join([f"[{a.severity}] {a.type}: {a.message}" for a in alerts])
    msg = MIMEText(body)
    msg["Subject"] = "DriftRadar Alerts"
    msg["From"] = settings.alert_email_from or settings.smtp_user
    msg["To"] = settings.alert_email_to

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        if settings.smtp_user and settings.smtp_password:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(msg)


def _load_batch_df(batch_id: int):
    path = os.path.join(settings.data_dir, f"batch_{batch_id}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _get_reference_batches(session: Session):
    first = session.query(Batch).order_by(Batch.timestamp.asc()).first()
    if not first:
        return []
    window_end = first.timestamp + timedelta(days=settings.reference_window_days)
    return session.query(Batch).filter(Batch.timestamp <= window_end).order_by(Batch.timestamp.asc()).all()


def process_batch(session: Session, batch_id: int, csv_path: str):
    df = pd.read_csv(csv_path)
    if settings.time_column in df.columns:
        df[settings.time_column] = pd.to_datetime(df[settings.time_column], errors="coerce")

    schema = infer_schema(df)
    features = ensure_features(session, schema)

    reference_batches = _get_reference_batches(session)
    ref_batch_id = reference_batches[0].id if reference_batches else None
    ref_stats = get_reference_stats(session, ref_batch_id)
    reference_df = None
    if reference_batches:
        dfs = []
        for b in reference_batches:
            df_b = _load_batch_df(b.id)
            if df_b is not None:
                dfs.append(df_b)
        if dfs:
            reference_df = pd.concat(dfs, ignore_index=True)
    quality = compute_quality(df, ref_stats)
    for qm in quality:
        feature_id = features[qm["feature"]].id if qm["feature"] else None
        session.add(QualityMetric(batch_id=batch_id, feature_id=feature_id, metric=qm["metric"], value=qm["value"]))

    rolling_stats = get_rolling_stats(session, batch_id)
    batch = session.get(Batch, batch_id)
    if not ref_stats:
        ref_stats = build_reference_stats(df, schema)
        batch.is_reference = True
        store_reference_stats(session, batch_id, ref_stats, features)
        train_model(df)
        reference_df = df
    else:
        if batch.timestamp <= reference_batches[0].timestamp + timedelta(days=settings.reference_window_days):
            batch.is_reference = True
            if reference_df is not None:
                ref_stats = build_reference_stats(reference_df, schema)
                store_reference_stats(session, ref_batch_id, ref_stats, features)
                train_model(reference_df)
    drift_metrics, current_stats = compute_drift(df, ref_stats, reference_df, rolling_stats)
    for dm in drift_metrics:
        session.add(DriftMetric(batch_id=batch_id, feature_id=features[dm["feature"]].id, metric=dm["metric"], value=dm["value"], comparison=dm["comparison"]))

    store_rolling_stats(session, batch_id, current_stats, features)

    model_eval = evaluate_model(df)
    if model_eval:
        for metric, value in model_eval.items():
            session.add(ModelMetric(batch_id=batch_id, metric=metric, value=value))

    alerts = create_alerts(session, batch_id, quality, drift_metrics)
    send_email_alerts(alerts)
    session.commit()


def create_batch(session: Session, timestamp: datetime, source: str, row_count: int, schema: dict):
    batch = Batch(timestamp=timestamp, source=source, row_count=row_count, schema_version="v1", schema_json=schema)
    session.add(batch)
    session.commit()
    session.refresh(batch)
    return batch
