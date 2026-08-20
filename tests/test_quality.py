"""Per-batch data quality."""

import pandas as pd
import pytest

from app.drift import build_reference_stats
from app.metrics import infer_schema
from app.quality import compute_quality


def reference_for(df):
    return build_reference_stats(df, infer_schema(df))


def only(metrics, metric, feature=None):
    found = [
        m for m in metrics if m["metric"] == metric and (feature is None or m["feature"] == feature)
    ]
    assert len(found) == 1, f"expected one {metric} for {feature}, got {len(found)}"
    return found[0]["value"]


def test_missing_rate_and_duplicates():
    df = pd.DataFrame({"a": [1, None, 1], "b": ["x", "x", "x"]})
    metrics = compute_quality(df, reference_stats=None)

    assert only(metrics, "missing_rate", "a") == pytest.approx(1 / 3)
    assert only(metrics, "duplicate_rate") > 0


def test_an_empty_batch_produces_no_metrics():
    assert compute_quality(pd.DataFrame({"a": []}), None) == [
        {"feature": "a", "metric": "missing_rate", "value": 0.0}
    ]


def test_unexpected_categories_are_counted_by_row():
    """This rate used to be unexpected labels over labels present, so one bad row
    in a thousand read as 0.33 while five hundred bad rows in the same thousand
    read as 0.50. The number moved the wrong way as the problem got worse."""
    reference = reference_for(pd.DataFrame({"c": ["a"] * 500 + ["b"] * 500}))

    one_bad = pd.DataFrame({"c": ["a"] * 500 + ["b"] * 499 + ["TYPO"]})
    many_bad = pd.DataFrame({"c": ["a"] * 500 + ["ZZZ"] * 500})

    rate_one = only(compute_quality(one_bad, reference), "unexpected_category_rate", "c")
    rate_many = only(compute_quality(many_bad, reference), "unexpected_category_rate", "c")

    assert rate_one == pytest.approx(0.001)
    assert rate_many == pytest.approx(0.5)
    assert rate_one < rate_many


def test_out_of_range_rate_counts_rows_outside_the_reference_range():
    reference = reference_for(pd.DataFrame({"x": [0.0, 10.0]}))
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0, -1.0, 99.0]})

    assert only(compute_quality(df, reference), "out_of_range_rate", "x") == pytest.approx(0.4)


def test_out_of_range_rate_ignores_nulls_rather_than_charging_them_twice():
    reference = reference_for(pd.DataFrame({"x": [0.0, 10.0]}))
    df = pd.DataFrame({"x": [5.0, None, 99.0]})
    metrics = compute_quality(df, reference)

    assert only(metrics, "missing_rate", "x") == pytest.approx(1 / 3)
    assert only(metrics, "out_of_range_rate", "x") == pytest.approx(0.5)


def test_a_datetime_column_gets_an_out_of_range_rate_instead_of_raising():
    """A datetime column is a numeric feature in the reference stats, and the
    out-of-range check called .astype(float) on it. That crashed the second
    upload of any CSV with a date in it."""
    reference = reference_for(pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-31"])}))
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-15", "2030-06-01"])})

    assert only(compute_quality(df, reference), "out_of_range_rate", "date") == pytest.approx(0.5)


def test_a_column_with_no_reference_gets_only_a_missing_rate():
    metrics = compute_quality(pd.DataFrame({"brand_new": [1, 2]}), {"other": {"type": "numeric"}})
    assert [m["metric"] for m in metrics] == ["duplicate_rate", "missing_rate"]
