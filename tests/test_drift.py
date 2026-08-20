"""Reference stats against current stats."""

import numpy as np
import pandas as pd
import pytest

from app.drift import build_current_stats, build_reference_stats, compute_drift
from app.metrics import infer_schema

from .conftest import values


def frame(seed=0, rows=2000, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "product": rng.choice(["A", "B", "C"], size=rows),
            "sales": 20 + shift + rng.normal(0, 3, rows),
        }
    )


def drift_for(reference: pd.DataFrame, current: pd.DataFrame, rolling=None):
    schema = infer_schema(reference)
    ref_stats = build_reference_stats(reference, schema)
    return compute_drift(current, ref_stats, reference, rolling)


def test_a_batch_with_a_date_column_does_not_raise():
    """Three separate call sites did .astype(float) on the datetime column, in
    build_current_stats, in ks_test, and in the out-of-range check. Any one of
    them ended the first upload with `TypeError: Cannot cast DatetimeArray to
    dtype float64`."""
    metrics, current = drift_for(frame(1), frame(2))

    assert current["date"]["type"] == "numeric"
    assert values(metrics, "psi", "date", "reference")


def test_no_drift_reads_as_no_drift():
    metrics, _ = drift_for(frame(1), frame(2))

    assert values(metrics, "psi", "sales", "reference")[0] < 0.05
    assert values(metrics, "jsd", "product", "reference")[0] < 0.01
    assert values(metrics, "ks_statistic", "sales", "reference")[0] < 0.1


def test_a_shifted_feature_reads_as_drift():
    metrics, _ = drift_for(frame(1), frame(2, shift=8.0))

    assert values(metrics, "psi", "sales", "reference")[0] > 0.2
    assert values(metrics, "ks_statistic", "sales", "reference")[0] > 0.5
    assert values(metrics, "ks_pvalue", "sales", "reference")[0] < 0.01


def test_a_reweighted_category_reads_as_drift():
    reference = pd.DataFrame({"product": ["A"] * 500 + ["B"] * 500})
    current = pd.DataFrame({"product": ["A"] * 900 + ["B"] * 100})
    metrics, _ = drift_for(reference, current)

    assert values(metrics, "jsd", "product", "reference")[0] > 0.1
    assert values(metrics, "chi_square_pvalue", "product", "reference")[0] < 0.001


def test_a_brand_new_category_reads_as_drift():
    reference = pd.DataFrame({"product": ["A"] * 500 + ["B"] * 500})
    current = pd.DataFrame({"product": ["A"] * 500 + ["NEW"] * 500})
    metrics, _ = drift_for(reference, current)

    assert values(metrics, "jsd", "product", "reference")[0] > 0.3


def test_the_current_histogram_uses_the_reference_bin_edges():
    """Two histograms with different edges are not comparable at all."""
    reference = frame(1)
    ref_stats = build_reference_stats(reference, infer_schema(reference))
    current = build_current_stats(frame(2, shift=50.0), ref_stats)

    assert current["sales"]["hist"]["bins"] == ref_stats["sales"]["hist"]["bins"]
    assert sum(current["sales"]["hist"]["counts"]) == pytest.approx(1.0)


def test_rolling_comparison_is_reported_separately_from_the_reference():
    reference = frame(1)
    ref_stats = build_reference_stats(reference, infer_schema(reference))
    previous = build_current_stats(frame(2), ref_stats)
    metrics, _ = compute_drift(frame(3, shift=8.0), ref_stats, reference, previous)

    assert values(metrics, "psi", "sales", "rolling")[0] > 0.2
    assert values(metrics, "psi", "sales", "reference")[0] > 0.2
    assert {m["comparison"] for m in metrics} == {"reference", "rolling"}


def test_a_column_the_reference_never_saw_is_skipped_not_crashed():
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [9.0, 9.0, 9.0]})
    metrics, current_stats = drift_for(reference, current)

    assert "b" in current_stats
    assert not [m for m in metrics if m["feature"] == "b"]


def test_a_column_that_disappeared_is_skipped_not_crashed():
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0], "gone": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    metrics, _ = drift_for(reference, current)

    assert not [m for m in metrics if m["feature"] == "gone"]


def test_a_column_that_changed_type_is_not_compared_as_if_it_had_not():
    reference = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"x": ["a", "b", "c"]})
    metrics, _ = drift_for(reference, current)

    assert not [m for m in metrics if m["metric"] in ("psi", "kl")]


def test_an_all_null_column_produces_no_metrics_and_no_exception():
    reference = pd.DataFrame({"x": [1.0, 2.0]})
    current = pd.DataFrame({"x": [None, None]}, dtype="float64")
    metrics, _ = drift_for(reference, current)

    assert not values(metrics, "psi", "x", "reference")


def test_a_boolean_column_is_compared_as_a_categorical():
    reference = pd.DataFrame({"flag": [True] * 500 + [False] * 500})
    current = pd.DataFrame({"flag": [True] * 100 + [False] * 900})
    metrics, _ = drift_for(reference, current)

    assert values(metrics, "jsd", "flag", "reference")[0] > 0.1
