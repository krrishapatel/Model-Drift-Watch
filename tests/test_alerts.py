"""The alert rules, with the metrics handed in directly.

Only three metrics used to be alerted on at all: missing_rate, psi and
ks_pvalue. Everything the categorical path computed, and every quality metric
past missing_rate, was stored and then never compared to anything, so a feature
could go completely wrong without the tool saying so.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db import Base
from app.ingestion import create_alerts
from app.models import Alert, Batch


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'alerts.db'}")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    s.add(
        Batch(
            id=1,
            timestamp=datetime(2024, 1, 1),
            source="test",
            row_count=1000,
            schema_version="v1",
            schema_json={},
        )
    )
    s.commit()
    yield s
    s.close()


def drift(feature, metric, value, comparison="reference"):
    return {"feature": feature, "metric": metric, "value": value, "comparison": comparison}


def quality(feature, metric, value):
    return {"feature": feature, "metric": metric, "value": value}


def messages(session, quality_metrics=(), drift_metrics=(), rows=1000):
    created = create_alerts(session, 1, list(quality_metrics), list(drift_metrics), rows)
    session.commit()
    assert len(created) == session.query(Alert).count()
    return [a.message for a in created]


@pytest.mark.parametrize(
    "metric,value,expected",
    [
        ("missing_rate", 0.5, "Missing rate"),
        ("out_of_range_rate", 0.5, "Out-of-range rate"),
        ("unexpected_category_rate", 0.5, "Unexpected category rate"),
        ("duplicate_rate", 0.5, "Duplicate row rate"),
    ],
)
def test_every_quality_metric_has_a_threshold(session, metric, value, expected):
    out = messages(session, quality_metrics=[quality("x", metric, value)])
    assert any(expected in m for m in out)


@pytest.mark.parametrize(
    "metric,value,expected",
    [
        ("psi", 5.0, "PSI"),
        ("jsd", 0.5, "Jensen-Shannon"),
        ("chi_square_pvalue", 1e-9, "Chi-square p-value"),
    ],
)
def test_every_drift_metric_has_a_threshold(session, metric, value, expected):
    assert any(expected in m for m in messages(session, drift_metrics=[drift("x", metric, value)]))


def test_a_quiet_batch_raises_nothing(session):
    assert (
        messages(
            session,
            quality_metrics=[quality("x", "missing_rate", 0.0)],
            drift_metrics=[drift("x", "psi", 0.01), drift("x", "jsd", 0.001)],
        )
        == []
    )


def test_the_alert_says_which_comparison_tripped(session):
    """ "PSI drift on sales" was the whole message, so a batch that had drifted
    from the reference and a batch that merely differed from the one before it
    produced identical text."""
    rolling = messages(session, drift_metrics=[drift("sales", "psi", 5.0, "rolling")])
    assert "rolling" in rolling[0]
    assert "sales" in rolling[0]


def test_ks_needs_both_a_small_pvalue_and_a_real_effect(session):
    """A p-value shrinks with the batch size for a shift of any size. Alerting on
    it alone means a large batch reports drift for a difference nobody would act
    on."""
    tiny_effect = messages(
        session,
        drift_metrics=[
            drift("sales", "ks_pvalue", 1e-8),
            drift("sales", "ks_statistic", 0.02),
        ],
    )
    assert tiny_effect == []


def test_ks_alerts_when_the_effect_is_real(session):
    both = messages(
        session,
        drift_metrics=[
            drift("sales", "ks_pvalue", 1e-8),
            drift("sales", "ks_statistic", 0.6),
        ],
    )
    assert any("KS on sales" in m for m in both)


def test_a_ks_statistic_with_no_pvalue_alerts_on_neither(session):
    assert messages(session, drift_metrics=[drift("sales", "ks_statistic", 0.9)]) == []


def test_a_batch_too_small_to_judge_is_said_so_and_not_alerted_on(session):
    out = messages(session, drift_metrics=[drift("sales", "psi", 5.0)], rows=50)

    assert not any("PSI" in m for m in out)
    assert any("below MIN_ROWS_FOR_DRIFT_ALERT" in m for m in out)


def test_a_small_batch_still_gets_its_quality_alerts(session):
    """Sampling noise is a reason to distrust a distribution comparison. It is not
    a reason to ignore a column that is half empty."""
    out = messages(session, quality_metrics=[quality("x", "missing_rate", 0.9)], rows=50)
    assert any("Missing rate" in m for m in out)


def test_thresholds_come_from_settings(session, monkeypatch):
    monkeypatch.setattr(settings, "psi_threshold", 10.0)
    assert messages(session, drift_metrics=[drift("sales", "psi", 5.0)]) == []


def test_the_time_column_never_raises_a_drift_alert(session):
    """Batch two covers later dates than batch one. That is what a new batch is.
    So the two date histograms share no bins and PSI comes out near its maximum
    on every upload no matter what the data looks like: measured at 12.434 on the
    two sample files, which differ only in when they were collected. A
    high-severity alert that fires every single time is one you learn to skip.
    """
    out = messages(
        session,
        drift_metrics=[
            drift("date", "psi", 12.434),
            drift("date", "ks_pvalue", 0.0),
            drift("date", "ks_statistic", 1.0),
            drift("date", "psi", 12.434, "rolling"),
            drift("sales", "psi", 5.0),
        ],
    )

    assert not any("date" in m for m in out)
    assert any("sales" in m for m in out)


def test_the_time_column_out_of_range_rate_never_alerts(session):
    """Same reason: every row of a newer batch is past the reference maximum, so
    the rate is 1.0 by construction."""
    out = messages(
        session,
        quality_metrics=[
            quality("date", "out_of_range_rate", 1.0),
            quality("sales", "out_of_range_rate", 0.5),
        ],
    )

    assert not any("date" in m for m in out)
    assert any("sales" in m for m in out)


def test_a_missing_date_still_alerts(session):
    """The exemption is for the metrics that move on their own. A null date is a
    broken row and has nothing to do with the batch being newer."""
    out = messages(session, quality_metrics=[quality("date", "missing_rate", 0.4)])
    assert any("Missing rate" in m and "date" in m for m in out)
