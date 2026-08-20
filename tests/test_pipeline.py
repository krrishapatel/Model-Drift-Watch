"""The API, end to end, with a real database and real statistics.

The evidence that this file needed to exist: the repository shipped a
drift_radar.db holding two batch rows and zero rows in every other table, which
is the footprint of create_batch committing and process_batch dying before its
own commit. Both times it was run, it produced no features, no metrics and no
alerts, and nothing in the repository noticed.
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import retail_csv, upload


def test_a_csv_can_be_ingested_and_becomes_the_reference(client):
    resp = upload(client, retail_csv(seed=1))
    assert resp.status_code == 200, resp.text
    assert resp.json()["row_count"] == 800
    assert resp.json()["mode"] in ("sync", "queued")

    batches = client.get("/batches").json()
    assert len(batches) == 1
    assert batches[0]["is_reference"] is True

    drift = client.get("/drift").json()
    quality = client.get("/quality").json()
    assert drift, "no drift metrics were stored"
    assert quality, "no quality metrics were stored"
    assert {"date", "product", "store", "sales"} <= {d["feature"] for d in drift}


def test_a_second_upload_is_measured_against_the_first(client):
    upload(client, retail_csv(seed=1))
    upload(client, retail_csv(seed=2))

    psi = [
        d
        for d in client.get("/drift", params={"batch_id": 2, "metric": "psi"}).json()
        if d["comparison"] == "reference"
    ]
    assert psi
    assert all(d["value"] < 0.2 for d in psi), psi


def test_the_time_column_stays_a_numeric_feature_across_batches(client):
    """Only the upload path parsed the time column. The reference set is rebuilt
    from CSVs reloaded off disk, where the dates were still strings, so they were
    coerced to null and the time feature quietly stopped being reported."""
    upload(client, retail_csv(seed=1))
    upload(client, retail_csv(seed=2))

    metrics = {
        d["metric"] for d in client.get("/drift", params={"batch_id": 2, "feature": "date"}).json()
    }
    assert "psi" in metrics
    assert "jsd" not in metrics  # not a categorical, and not silently dropped


def test_two_undrifted_batches_raise_no_drift_alert(client):
    upload(client, retail_csv(seed=1))
    upload(client, retail_csv(seed=2))

    alerts = client.get("/alerts").json()
    assert [a for a in alerts if a["type"] == "drift"] == []


def test_a_drifted_batch_raises_an_alert_naming_the_feature_and_the_comparison(client):
    upload(client, retail_csv(seed=1), role="reference")
    upload(client, retail_csv(seed=2, sales_shift=12.0), role="current")

    drift_alerts = [a for a in client.get("/alerts").json() if a["type"] == "drift"]
    assert drift_alerts
    assert any("sales" in a["message"] and "reference" in a["message"] for a in drift_alerts)


def test_a_current_batch_never_joins_the_reference(client):
    """Uploading a backlog in one sitting put every batch inside the reference
    window, so a drifted batch became part of the reference it was supposed to be
    compared against."""
    upload(client, retail_csv(seed=1), role="reference")
    upload(client, retail_csv(seed=2, sales_shift=12.0), role="current")

    batches = {b["id"]: b for b in client.get("/batches").json()}
    assert batches[1]["is_reference"] is True
    assert batches[2]["is_reference"] is False
    assert batches[2]["role"] == "current"


def test_a_small_batch_is_measured_but_not_alerted_on(client):
    """PSI on 10 bins has a null expectation near 2*(bins-1)/rows. Measured over
    400 drift-free pairs: at 100 rows, 57% came out above the 0.2 threshold. The
    metric is still computed and stored; it just is not called drift."""
    upload(client, retail_csv(seed=1, rows=100), role="reference")
    upload(client, retail_csv(seed=2, rows=100), role="current")

    assert client.get("/drift", params={"batch_id": 2, "metric": "psi"}).json()

    alerts = client.get("/alerts").json()
    assert [a for a in alerts if a["type"] == "drift"] == []
    assert any("below MIN_ROWS_FOR_DRIFT_ALERT" in a["message"] for a in alerts)


def test_missing_values_raise_a_quality_alert(client):
    rng = np.random.default_rng(4)
    sales = pd.Series(20 + rng.normal(0, 3, 800))
    sales[:200] = None
    upload(client, retail_csv(seed=1, sales=sales))

    quality_alerts = [a for a in client.get("/alerts").json() if a["type"] == "quality"]
    assert any("Missing rate" in a["message"] and "sales" in a["message"] for a in quality_alerts)


def test_an_unexpected_category_raises_a_quality_alert(client):
    upload(client, retail_csv(seed=1), role="reference")

    rng = np.random.default_rng(5)
    product = rng.choice(["A", "B", "C"], size=800)
    product[:100] = "UNLISTED"
    upload(client, retail_csv(seed=2, product=product), role="current")

    messages = [a["message"] for a in client.get("/alerts").json()]
    assert any("Unexpected category rate" in m and "product" in m for m in messages)


def test_values_outside_the_reference_range_raise_a_quality_alert(client):
    upload(client, retail_csv(seed=1), role="reference")

    rng = np.random.default_rng(6)
    sales = pd.Series(20 + rng.normal(0, 3, 800))
    sales[:80] = 5000.0
    upload(client, retail_csv(seed=2, sales=sales), role="current")

    messages = [a["message"] for a in client.get("/alerts").json()]
    assert any("Out-of-range rate" in m and "sales" in m for m in messages)


def test_model_health_is_recorded_for_every_batch(client):
    upload(client, retail_csv(seed=1))
    upload(client, retail_csv(seed=2))

    health = client.get("/model_health").json()
    assert {m["metric"] for m in health} >= {"mae", "mape"}
    assert {m["batch_id"] for m in health} == {1, 2}


def test_something_that_is_not_a_csv_is_a_400_not_a_500(client):
    resp = client.post("/ingest", files={"file": ("x.csv", b"\x00\x01\x02\x03", "text/csv")})
    assert resp.status_code == 400
    assert client.get("/batches").json() == []


def test_an_empty_csv_is_a_400(client):
    resp = client.post("/ingest", files={"file": ("x.csv", b"a,b\n", "text/csv")})
    assert resp.status_code == 400


def test_an_oversized_upload_is_a_413(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 0)
    resp = upload(client, retail_csv(seed=1))

    assert resp.status_code == 413
    assert client.get("/batches").json() == []


def test_an_unknown_role_is_a_400(client):
    resp = upload(client, retail_csv(seed=1), role="whatever")
    assert resp.status_code == 400
    assert client.get("/batches").json() == []


def test_reprocessing_a_batch_does_not_duplicate_its_rolling_stats(client):
    """store_reference_stats cleared the batch's old rows first and
    store_rolling_stats did not, so a reprocessed batch kept both versions."""
    import os

    from app.config import settings
    from app.db import SessionLocal
    from app.ingestion import process_batch
    from app.models import RollingStat

    upload(client, retail_csv(seed=1))

    session = SessionLocal()
    try:
        before = session.query(RollingStat).count()
        assert before > 0
        process_batch(session, 1, os.path.join(settings.data_dir, "batch_1.csv"))
        assert session.query(RollingStat).count() == before
    finally:
        session.close()


@pytest.mark.parametrize("endpoint", ["/batches", "/drift", "/quality", "/model_health", "/alerts"])
def test_every_endpoint_answers_on_an_empty_database(client, endpoint):
    resp = client.get(endpoint)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.parametrize("endpoint", ["/drift", "/quality"])
def test_the_metric_endpoints_can_be_filtered_by_feature(client, endpoint):
    """Without a feature filter the dashboard has to pull every metric for every
    column of every batch and sort it out in the browser."""
    upload(client, retail_csv(seed=1))
    upload(client, retail_csv(seed=2))

    unfiltered = client.get(endpoint).json()
    filtered = client.get(endpoint, params={"feature": "sales"}).json()

    assert filtered
    assert len(filtered) < len(unfiltered)
    assert {row["feature"] for row in filtered} == {"sales"}
