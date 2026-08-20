import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue

from .config import settings
from .db import Base, SessionLocal, engine
from .ingestion import ROLES, create_batch, read_csv, save_raw_file
from .jobs import process_batch_job
from .metrics import infer_schema
from .models import Alert, Batch, DriftMetric, Feature, ModelMetric, QualityMetric

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="DriftRadar", lifespan=lifespan)

# Anchored to the repo root, not the working directory. StaticFiles raises at
# import time if the directory is not there, so with a relative path the app
# could not be imported from anywhere else.
FRONTEND_DIR = settings.frontend_dir
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(f"{FRONTEND_DIR}/index.html")


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    source: str = "upload",
    role: str = Query(default="auto", description="auto, reference, or current"),
):
    if role not in ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {', '.join(ROLES)}")

    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is larger than MAX_UPLOAD_MB ({settings.max_upload_mb} MB).",
        )

    # A CSV that pandas cannot parse is the caller's problem, not a server error.
    try:
        df = read_csv(io.BytesIO(data))
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as err:
        raise HTTPException(status_code=400, detail=f"Could not read the upload as CSV: {err}")
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV has no rows.")

    schema = infer_schema(df)
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    session = SessionLocal()
    try:
        batch = create_batch(session, timestamp, source, len(df), schema, role=role)
        csv_path = save_raw_file(data, batch.id)
        try:
            redis_conn = Redis.from_url(settings.redis_url)
            redis_conn.ping()
            Queue(connection=redis_conn).enqueue(process_batch_job, batch.id, csv_path)
            mode = "queued"
        except RedisError as err:
            # No queue available, so do the work in the request. Catches every
            # redis failure, not only a refused connection: a timeout or an auth
            # error used to escape as a 500 with the batch row already committed.
            logger.info("redis unavailable (%s); processing batch %s inline", err, batch.id)
            process_batch_job(batch.id, csv_path)
            mode = "sync"
        return {"batch_id": batch.id, "row_count": len(df), "role": role, "mode": mode}
    finally:
        session.close()


@app.get("/batches")
def list_batches():
    session = SessionLocal()
    try:
        batches = session.query(Batch).order_by(Batch.id.desc()).all()
        return [
            {
                "id": b.id,
                "timestamp": b.timestamp,
                "source": b.source,
                "row_count": b.row_count,
                "schema_version": b.schema_version,
                "role": b.role or "auto",
                "is_reference": b.is_reference,
            }
            for b in batches
        ]
    finally:
        session.close()


@app.get("/drift")
def get_drift(
    feature: str | None = Query(default=None),
    batch_id: int | None = Query(default=None),
    metric: str | None = Query(default=None),
):
    session = SessionLocal()
    try:
        query = session.query(DriftMetric, Feature).join(
            Feature, DriftMetric.feature_id == Feature.id
        )
        if feature:
            query = query.filter(Feature.name == feature)
        if batch_id:
            query = query.filter(DriftMetric.batch_id == batch_id)
        if metric:
            query = query.filter(DriftMetric.metric == metric)
        rows = query.order_by(DriftMetric.batch_id.asc(), DriftMetric.id.asc()).all()
        return [
            {
                "batch_id": dm.batch_id,
                "feature": f.name,
                "metric": dm.metric,
                "value": dm.value,
                "comparison": dm.comparison,
            }
            for dm, f in rows
        ]
    finally:
        session.close()


@app.get("/quality")
def get_quality(
    feature: str | None = Query(default=None),
    batch_id: int | None = Query(default=None),
    metric: str | None = Query(default=None),
):
    session = SessionLocal()
    try:
        query = session.query(QualityMetric, Feature).outerjoin(
            Feature, QualityMetric.feature_id == Feature.id
        )
        if feature:
            query = query.filter(Feature.name == feature)
        if batch_id:
            query = query.filter(QualityMetric.batch_id == batch_id)
        if metric:
            query = query.filter(QualityMetric.metric == metric)
        rows = query.order_by(QualityMetric.batch_id.asc(), QualityMetric.id.asc()).all()
        return [
            {
                "batch_id": qm.batch_id,
                "feature": f.name if f else None,
                "metric": qm.metric,
                "value": qm.value,
            }
            for qm, f in rows
        ]
    finally:
        session.close()


@app.get("/model_health")
def get_model_health():
    session = SessionLocal()
    try:
        rows = session.query(ModelMetric).order_by(ModelMetric.batch_id.asc()).all()
        return [{"batch_id": mm.batch_id, "metric": mm.metric, "value": mm.value} for mm in rows]
    finally:
        session.close()


@app.get("/alerts")
def get_alerts(
    batch_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
):
    """The alerts. They were written to the database and never read back out."""
    session = SessionLocal()
    try:
        query = session.query(Alert)
        if batch_id:
            query = query.filter(Alert.batch_id == batch_id)
        if status:
            query = query.filter(Alert.status == status)
        rows = query.order_by(Alert.batch_id.desc(), Alert.id.desc()).all()
        return [
            {
                "id": a.id,
                "batch_id": a.batch_id,
                "type": a.type,
                "severity": a.severity,
                "status": a.status,
                "message": a.message,
                "created_at": a.created_at,
            }
            for a in rows
        ]
    finally:
        session.close()
