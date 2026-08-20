"""The statistics, on their own."""

import numpy as np
import pandas as pd
import pytest

from app.metrics import (
    align_categories,
    categorical_counts,
    chi_square,
    hist_on_bins,
    infer_schema,
    js_divergence,
    ks_test,
    numeric_hist,
    psi,
    to_float_series,
)


def test_psi_and_jsd_are_zero_for_identical_distributions():
    a = np.array([0.2, 0.3, 0.5])
    assert psi(a, a) == 0.0
    assert js_divergence(a, a) == 0.0


def test_a_boolean_column_is_typed_boolean():
    """is_numeric_dtype is True for a bool column, so with the checks in the
    other order every boolean was typed numeric and the boolean branch was
    unreachable."""
    schema = infer_schema(pd.DataFrame({"flag": [True, False, True]}))
    assert schema["columns"][0]["type"] == "boolean"


def test_column_types():
    df = pd.DataFrame(
        {
            "n": [1.0, 2.0],
            "s": ["a", "b"],
            "d": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "b": [True, False],
        }
    )
    got = {c["name"]: c["type"] for c in infer_schema(df)["columns"]}
    assert got == {"n": "numeric", "s": "categorical", "d": "datetime", "b": "boolean"}


def test_a_datetime_column_converts_to_epoch_seconds():
    """`series.astype(float)` raises TypeError on a datetime column. Four call
    sites did exactly that, and a datetime column is stored as a numeric feature,
    so the first upload always hit one of them."""
    s = pd.Series(pd.to_datetime(["1970-01-01", "1970-01-02"]))
    assert list(to_float_series(s)) == [0.0, 86400.0]


def test_unparseable_values_become_null_not_an_exception():
    assert list(to_float_series(pd.Series(["1.5", "oops", None]))) == [1.5]


def test_psi_sees_a_batch_that_moved_out_of_the_reference_range():
    """np.histogram drops out-of-range rows, and the survivors were renormalised
    to sum to 1, so the drifted rows were discarded and the rest looked
    unchanged. Measured before the fix on this exact input: PSI 0.0165, against
    a threshold of 0.2."""
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(20, 3, 4000))
    ref = numeric_hist(reference, bins=10)

    moved = pd.Series(np.concatenate([rng.normal(20, 3, 3000), rng.normal(200, 5, 1000)]))
    cur = hist_on_bins(moved, ref["bins"])

    assert psi(np.array(ref["counts"]), np.array(cur["counts"])) > 0.2


def test_clipping_puts_out_of_range_rows_in_the_edge_bins():
    """The rows are kept, not dropped: a quarter of them left the range, so a
    quarter of the mass has to land in the top bin."""
    ref = numeric_hist(pd.Series(np.linspace(0, 10, 1000)), bins=10)
    cur = hist_on_bins(pd.Series([5.0] * 750 + [10_000.0] * 250), ref["bins"])
    counts = np.array(cur["counts"])

    assert counts.sum() == pytest.approx(1.0)
    assert counts[-1] == pytest.approx(0.25)


def test_a_reference_range_with_no_width_does_not_divide_by_zero():
    hist = numeric_hist(pd.Series([4.0, 4.0, 4.0]), bins=10)
    assert sum(hist["counts"]) == pytest.approx(1.0)
    assert hist_on_bins(pd.Series([4.0]), hist["bins"])["counts"][0] == pytest.approx(1.0)


def test_an_empty_column_gives_empty_stats_rather_than_raising():
    assert numeric_hist(pd.Series([None, None], dtype="float64")) == {"bins": [], "counts": []}
    assert categorical_counts(pd.Series([None, None], dtype="object")) == {
        "categories": [],
        "counts": [],
    }
    assert hist_on_bins(pd.Series([], dtype="float64"), [0, 1, 2])["counts"] == []


def test_chi_square_grows_with_the_number_of_rows():
    """Handed proportions, scipy's chisquare divides the statistic by the sample
    size. Measured before the fix: a 70/30 split against a 50/50 reference
    returned 0.16 on 10 rows and 0.16 on 100,000."""
    ref = np.array([0.5, 0.5])
    cur = np.array([0.7, 0.3])

    small_stat, small_p = chi_square(ref, cur, n_rows=10)
    large_stat, large_p = chi_square(ref, cur, n_rows=100_000)

    assert large_stat > small_stat * 1000
    assert small_p > 0.05  # 10 rows cannot establish this shift
    assert large_p < 1e-10  # 100,000 rows can


def test_chi_square_is_flat_when_nothing_moved():
    props = np.array([0.5, 0.3, 0.2])
    stat, pvalue = chi_square(props, props, n_rows=5000)
    assert stat == pytest.approx(0.0, abs=1e-6)
    assert pvalue == pytest.approx(1.0)


def test_chi_square_handles_an_empty_reference():
    assert chi_square(np.array([]), np.array([]), n_rows=100) == (0.0, 1.0)


def test_ks_returns_the_statistic_as_well_as_the_pvalue():
    """The p-value alone is not enough to threshold on: it shrinks as the batch
    grows, for shifts of any size. The statistic does not."""
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0, 1, 20_000))
    b = pd.Series(rng.normal(0.02, 1, 20_000))

    statistic, pvalue = ks_test(a, b)

    assert pvalue < 0.05  # "not chance"
    assert statistic < 0.1  # and yet the two distributions nearly coincide


def test_ks_on_a_datetime_column():
    a = pd.Series(pd.date_range("2024-01-01", periods=500, freq="D"))
    b = pd.Series(pd.date_range("2024-01-01", periods=500, freq="D"))
    statistic, pvalue = ks_test(a, b)
    assert statistic == 0.0
    assert pvalue == pytest.approx(1.0)


def test_ks_on_an_empty_column_reports_no_difference():
    empty = pd.Series([], dtype="float64")
    assert ks_test(empty, pd.Series([1.0, 2.0])) == (0.0, 1.0)


def test_align_categories_covers_both_sides():
    ref = {"categories": ["a", "b"], "counts": [0.5, 0.5]}
    cur = {"categories": ["b", "z"], "counts": [0.5, 0.5]}
    ref_aligned, cur_aligned = align_categories(ref, cur)

    assert list(ref_aligned) == [0.5, 0.5, 0.0]
    assert list(cur_aligned) == [0.0, 0.5, 0.5]
