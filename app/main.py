from datetime import datetime
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue

from .config import settings
from .db import Base, engine, SessionLocal
from .ingestion import create_batch, save_raw_file
from .jobs import process_batch_job
from .metrics import infer_schema
from .models import Batch, Feature, DriftMetric, QualityMetric, ModelMetric

app = FastAPI(title="DriftRadar")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


app.mount("/static", StaticFiles(directory="./frontend"), name="static")


@app.get("/")
def root():
    return FileResponse("./frontend/index.html")


@app.post("/ingest")
def ingest(file: UploadFile = File(...), source: str = "upload"):
    data = file.file.read()
    df = pd.read_csv(io.BytesIO(data))
    row_count = len(df)
    schema = infer_schema(df)

    timestamp = datetime.utcnow()

    session = SessionLocal()
    try:
        batch = create_batch(session, timestamp, source, row_count, schema)
        csv_path = save_raw_file(data, batch.id)
        try:
            redis_conn = Redis.from_url(settings.redis_url)
            redis_conn.ping()
            q = Queue(connection=redis_conn)
            q.enqueue(process_batch_job, batch.id, csv_path)
            mode = "queued"
        except RedisConnectionError:
            process_batch_job(batch.id, csv_path)
            mode = "sync"
        return {"batch_id": batch.id, "row_count": row_count, "mode": mode}
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
                "is_reference": b.is_reference,
            }
            for b in batches
        ]
    finally:
        session.close()


@app.get("/drift")
def get_drift(feature: str | None = Query(default=None)):
    session = SessionLocal()
    try:
        query = session.query(DriftMetric, Feature).join(Feature, DriftMetric.feature_id == Feature.id)
        if feature:
            query = query.filter(Feature.name == feature)
        rows = query.all()
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
def get_quality(batch_id: int | None = Query(default=None)):
    session = SessionLocal()
    try:
        query = session.query(QualityMetric, Feature).outerjoin(Feature, QualityMetric.feature_id == Feature.id)
        if batch_id:
            query = query.filter(QualityMetric.batch_id == batch_id)
        rows = query.all()
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
        rows = session.query(ModelMetric).all()
        return [
            {
                "batch_id": mm.batch_id,
                "metric": mm.metric,
                "value": mm.value,
            }
            for mm in rows
        ]
    finally:
        session.close()
