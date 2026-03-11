from sqlalchemy.orm import Session

from .db import SessionLocal
from .ingestion import process_batch


def process_batch_job(batch_id: int, csv_path: str):
    session: Session = SessionLocal()
    try:
        process_batch(session, batch_id, csv_path)
    finally:
        session.close()
