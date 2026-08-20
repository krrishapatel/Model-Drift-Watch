"""Everything that happens to a batch after it lands: stats, metrics, alerts.

Which batches form the reference is decided here, and it is worth being explicit
about it. By default the reference is every batch uploaded within
REFERENCE_WINDOW_DAYS of the first one, measured by upload time, which suits
batches that arrive on a schedule. It does not suit uploading a backlog in one
sitting: every batch lands inside the window, so every batch joins the reference,
including a batch that has drifted. Pass `role=current` on the upload to keep a
batch out of the reference, or `role=reference` to put it in regardless of when
it arrived.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .config import settings
from .drift import build_reference_stats, compute_drift
from .metrics import infer_schema
from .model_health import evaluate_model, train_model
from .models import (
    Alert,
    Batch,
    DriftMetric,
    Feature,
    ModelMetric,
    QualityMetric,
    ReferenceStat,
    RollingStat,
)
from .quality import compute_quality

logger = logging.getLogger(__name__)

ROLES = ("auto", "reference", "current")


def read_csv(path_or_buf) -> pd.DataFrame:
    """The one way a batch is read.

    The time column is parsed here so that a batch reloaded from disk is typed
    the same as the batch that was just uploaded. When only the upload path
    parsed it, the reference stats rebuilt from reloaded CSVs saw the dates as
    strings, coerced them all to null, and dropped the time feature from the
    drift report without saying anything.
    """
    df = pd.read_csv(path_or_buf)
    if settings.time_column in df.columns:
        df[settings.time_column] = pd.to_datetime(df[settings.time_column], errors="coerce")
    return df


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
    rows = (
        session.query(ReferenceStat, Feature)
        .join(Feature, Feature.id == ReferenceStat.feature_id)
        .filter(ReferenceStat.batch_id == ref_batch_id)
        .all()
    )
    for row, feature in rows:
        stats[feature.name] = row.stats_json
    return stats


def get_rolling_stats(session: Session, current_batch_id: int):
    last_batch = (
        session.query(Batch).filter(Batch.id < current_batch_id).order_by(Batch.id.desc()).first()
    )
    if not last_batch:
        return None
    stats = {}
    rows = (
        session.query(RollingStat, Feature)
        .join(Feature, Feature.id == RollingStat.feature_id)
        .filter(RollingStat.batch_id == last_batch.id)
        .all()
    )
    for row, feature in rows:
        stats[feature.name] = row.stats_json
    return stats


def store_reference_stats(session: Session, batch_id: int, stats: dict, features: dict):
    session.query(ReferenceStat).filter(ReferenceStat.batch_id == batch_id).delete()
    for name, payload in stats.items():
        session.add(
            ReferenceStat(batch_id=batch_id, feature_id=features[name].id, stats_json=payload)
        )


def store_rolling_stats(session: Session, batch_id: int, stats: dict, features: dict):
    # Cleared first, like the reference stats are. Without this, reprocessing a
    # batch left both versions in the table.
    session.query(RollingStat).filter(RollingStat.batch_id == batch_id).delete()
    for name, payload in stats.items():
        session.add(
            RollingStat(batch_id=batch_id, feature_id=features[name].id, stats_json=payload)
        )


# metric name -> (threshold setting, direction, severity, human label)
# ">" alerts when the value is above the threshold, "<" when it is below.
QUALITY_RULES = {
    "missing_rate": ("missing_rate_threshold", ">", "medium", "Missing rate"),
    "out_of_range_rate": ("out_of_range_rate_threshold", ">", "medium", "Out-of-range rate"),
    "unexpected_category_rate": (
        "unexpected_category_rate_threshold",
        ">",
        "medium",
        "Unexpected category rate",
    ),
    "duplicate_rate": ("duplicate_rate_threshold", ">", "low", "Duplicate row rate"),
}

DRIFT_RULES = {
    "psi": ("psi_threshold", ">", "high", "PSI"),
    "jsd": ("jsd_threshold", ">", "high", "Jensen-Shannon distance"),
    "chi_square_pvalue": (
        "chi_square_pvalue_threshold",
        "<",
        "medium",
        "Chi-square p-value",
    ),
}


def _trips(value: float, threshold: float, direction: str) -> bool:
    return value > threshold if direction == ">" else value < threshold


# The time column moves by construction, so alerting on it is a guaranteed false
# alarm on every upload. Consecutive batches cover different date ranges, which
# means the reference histogram and the current one share no bins at all.
# Measured on the two sample files, which differ only in when they were
# collected: PSI 12.4, KS statistic 1.000, out-of-range rate 1.000. A monitor
# that reports "this batch is more recent than the last one" at high severity
# every time teaches you to ignore it. The metrics are still computed and stored
# so the dashboard can show the range moving; they raise nothing.
#
# Missing dates are a real problem and still alert. So does an unexpected
# category, since the time column is never categorical.
TIME_COLUMN_EXEMPT = ("out_of_range_rate",)


def _skip_time_column(feature: str | None, metric: str, exempt=None) -> bool:
    if feature != settings.time_column:
        return False
    return exempt is None or metric in exempt


def create_alerts(
    session: Session,
    batch_id: int,
    quality_metrics: list,
    drift_metrics: list,
    row_count: int,
):
    alerts = []

    for qm in quality_metrics:
        rule = QUALITY_RULES.get(qm["metric"])
        if not rule:
            continue
        setting, direction, severity, label = rule
        if _skip_time_column(qm["feature"], qm["metric"], TIME_COLUMN_EXEMPT):
            continue
        if _trips(qm["value"], getattr(settings, setting), direction):
            where = f" for {qm['feature']}" if qm["feature"] else ""
            alerts.append(
                Alert(
                    batch_id=batch_id,
                    type="quality",
                    severity=severity,
                    status="open",
                    message=f"{label} high{where}: {qm['value']:.3f}",
                )
            )

    # Under the null, PSI on 10 bins runs at roughly 2*(bins-1)/rows. Measured
    # over 400 pairs drawn from one normal distribution: at 100 rows a median of
    # 0.22 and 57% of comparisons above the 0.2 threshold; at 300 rows, 2%; at
    # 1000 rows, none. So on a small batch the alert is mostly reporting its own
    # sampling noise. The metrics are still stored, and still plotted.
    if row_count < settings.min_rows_for_drift_alert:
        alerts.append(
            Alert(
                batch_id=batch_id,
                type="info",
                severity="low",
                status="open",
                message=(
                    f"{row_count} rows is below MIN_ROWS_FOR_DRIFT_ALERT "
                    f"({settings.min_rows_for_drift_alert}); drift metrics were computed "
                    "but not alerted on, since at this size they are mostly sampling noise."
                ),
            )
        )
    else:
        by_feature = {}
        for dm in drift_metrics:
            by_feature.setdefault((dm["feature"], dm["comparison"]), {})[dm["metric"]] = dm["value"]

        for (feature, comparison), values in by_feature.items():
            # Every drift metric on the time column, not a subset of them.
            if _skip_time_column(feature, "", exempt=None):
                continue
            for metric, value in sorted(values.items()):
                rule = DRIFT_RULES.get(metric)
                if not rule:
                    continue
                setting, direction, severity, label = rule
                if _trips(value, getattr(settings, setting), direction):
                    alerts.append(
                        Alert(
                            batch_id=batch_id,
                            type="drift",
                            severity=severity,
                            status="open",
                            # Which comparison it was. The message used to say
                            # only "PSI drift on sales", so a drift-free batch
                            # that tripped the rolling comparison looked
                            # identical to one that had left the reference.
                            message=(
                                f"{label} on {feature} vs {comparison}: {value:.3f} "
                                f"(threshold {getattr(settings, setting)})"
                            ),
                        )
                    )

            # Both KS gates, or neither. A p-value alone says the shift is not
            # chance, which stops being interesting once the batch is large.
            pvalue = values.get("ks_pvalue")
            statistic = values.get("ks_statistic")
            if pvalue is not None and statistic is not None:
                if (
                    pvalue < settings.ks_pvalue_threshold
                    and statistic > settings.ks_statistic_threshold
                ):
                    alerts.append(
                        Alert(
                            batch_id=batch_id,
                            type="drift",
                            severity="medium",
                            status="open",
                            message=(
                                f"KS on {feature} vs {comparison}: statistic "
                                f"{statistic:.3f}, p={pvalue:.4f}"
                            ),
                        )
                    )

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
    return read_csv(path)


def _get_reference_batches(session: Session):
    """The batches the reference is built from, oldest first.

    A batch uploaded with role=current is never one of them, however early it
    arrived.
    """
    candidates = (
        session.query(Batch)
        .filter(or_(Batch.role.is_(None), Batch.role != "current"))
        .order_by(Batch.timestamp.asc())
        .all()
    )
    if not candidates:
        return []
    window_end = candidates[0].timestamp + timedelta(days=settings.reference_window_days)
    return [b for b in candidates if b.timestamp <= window_end]


def process_batch(session: Session, batch_id: int, csv_path: str):
    df = read_csv(csv_path)
    schema = infer_schema(df)
    features = ensure_features(session, schema)
    batch = session.get(Batch, batch_id)
    role = batch.role or "auto"

    reference_batches = [b for b in _get_reference_batches(session) if b.id != batch_id]
    ref_batch_id = reference_batches[0].id if reference_batches else None
    ref_stats = get_reference_stats(session, ref_batch_id)

    reference_df = None
    if reference_batches:
        dfs = [d for d in (_load_batch_df(b.id) for b in reference_batches) if d is not None]
        if dfs:
            reference_df = pd.concat(dfs, ignore_index=True)

    quality = compute_quality(df, ref_stats)
    for qm in quality:
        feature_id = features[qm["feature"]].id if qm["feature"] else None
        session.add(
            QualityMetric(
                batch_id=batch_id,
                feature_id=feature_id,
                metric=qm["metric"],
                value=qm["value"],
            )
        )

    rolling_stats = get_rolling_stats(session, batch_id)

    # What this batch is measured against: the reference as it stood before this
    # batch arrived. A batch that joins the reference updates it afterwards, so
    # it does not dilute its own comparison. Only the very first batch has
    # nothing else to compare to.
    compare_stats = ref_stats

    if role == "current":
        joins_reference = False
    elif role == "reference" or not ref_stats:
        joins_reference = True
    else:
        window_end = reference_batches[0].timestamp + timedelta(days=settings.reference_window_days)
        joins_reference = batch.timestamp <= window_end

    if joins_reference:
        batch.is_reference = True
        if reference_df is not None:
            combined = pd.concat([reference_df, df], ignore_index=True)
        else:
            combined = df
        ref_stats = build_reference_stats(combined, schema)
        store_reference_stats(session, ref_batch_id or batch_id, ref_stats, features)
        train_model(combined)
        if compare_stats is None:
            compare_stats = ref_stats

    drift_metrics, current_stats = compute_drift(
        df, compare_stats or {}, reference_df, rolling_stats
    )
    for dm in drift_metrics:
        session.add(
            DriftMetric(
                batch_id=batch_id,
                feature_id=features[dm["feature"]].id,
                metric=dm["metric"],
                value=dm["value"],
                comparison=dm["comparison"],
            )
        )

    store_rolling_stats(session, batch_id, current_stats, features)

    model_eval = evaluate_model(df)
    if model_eval:
        for metric, value in model_eval.items():
            session.add(ModelMetric(batch_id=batch_id, metric=metric, value=value))

    alerts = create_alerts(session, batch_id, quality, drift_metrics, len(df))
    session.commit()
    # After the commit, and never allowed to undo it. An SMTP server that is
    # down is not a reason to lose the batch that was just measured.
    try:
        send_email_alerts(alerts)
    except OSError as err:
        logger.warning("could not send alert email: %s", err)


def create_batch(
    session: Session,
    timestamp: datetime,
    source: str,
    row_count: int,
    schema: dict,
    role: str = "auto",
):
    batch = Batch(
        timestamp=timestamp,
        source=source,
        row_count=row_count,
        schema_version="v1",
        schema_json=schema,
        role=role,
    )
    session.add(batch)
    session.commit()
    session.refresh(batch)
    return batch
