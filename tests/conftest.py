"""Shared fixtures.

Nothing here is mocked. The statistics are the product, so a test that stubbed
them out would be testing nothing, and the two test files that existed before
this one could not even be collected: they imported `drift_radar.app.metrics`,
a package that does not exist, and `ks_from_hist`, a function that does not
exist. With no CI to run them, that went unnoticed.

Every test gets its own database, upload directory, and model file, so ordering
cannot matter.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "raw"))
    monkeypatch.setattr(settings, "artifact_path", str(tmp_path / "models" / "model.joblib"))
    monkeypatch.setattr(settings, "alert_email_enabled", False)
    yield


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """A database of its own, wired into db.py and everything that imported it."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import db, jobs, main
    from app.models import Base as _  # noqa: F401  (ensures models are registered)

    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db.Base.metadata.create_all(bind=engine)
    for module in (db, jobs, main):
        monkeypatch.setattr(module, "SessionLocal", Session, raising=False)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(main, "engine", engine)
    return Session


@pytest.fixture
def client(fresh_db):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def retail_csv(seed: int = 1, rows: int = 800, sales_shift: float = 0.0, **overrides) -> bytes:
    """A retail batch shaped like the one scripts/seed_sample.py writes.

    Default row count is above MIN_ROWS_FOR_DRIFT_ALERT so drift alerts are live
    unless a test deliberately goes smaller.
    """
    rng = np.random.default_rng(seed)
    frame = {
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
        "product": rng.choice(["A", "B", "C"], size=rows),
        "store": rng.choice(["north", "south"], size=rows),
        "sales": 20 + sales_shift + 5 * (rng.random(rows) < 0.3) + rng.normal(0, 3, rows),
    }
    frame.update(overrides)
    return pd.DataFrame(frame).to_csv(index=False).encode()


def upload(client, data: bytes, name: str = "batch.csv", **params):
    return client.post("/ingest", files={"file": (name, data, "text/csv")}, params=params or None)


def values(rows, metric, feature=None, comparison=None):
    out = [r for r in rows if r["metric"] == metric]
    if feature is not None:
        out = [r for r in out if r["feature"] == feature]
    if comparison is not None:
        out = [r for r in out if r["comparison"] == comparison]
    return [r["value"] for r in out]
