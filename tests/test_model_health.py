"""Training and scoring the baseline model."""

import numpy as np
import pandas as pd

from app.model_health import _prepare_features, evaluate_model, train_model


def retail(rows=400, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    product = rng.choice(["A", "B", "C"], size=rows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "product": product,
            "sales": 20 + shift + 5 * (product == "A") + rng.normal(0, 1, rows),
        }
    )


def test_the_raw_time_column_does_not_reach_the_model():
    """Left in, it arrived at sklearn as a datetime64 column, which raises
    DTypePromotionError, or, when the batch had been reloaded from CSV and the
    dates were still strings, as one indicator column per distinct date. The
    parts worth learning from are extracted first."""
    X, y = _prepare_features(retail())

    assert "date" not in X.columns
    assert {"dow", "month", "day"} <= set(X.columns)
    assert not [c for c in X.columns if c.startswith("date_")]
    assert y is not None


def test_training_a_batch_with_a_date_column_succeeds():
    """This was the first thing to fail, on the first upload, every time."""
    assert train_model(retail()) is not None


def test_a_string_date_column_does_not_become_one_column_per_date():
    """A batch reloaded from disk with no dtype parsing, which is how the
    reference set used to be assembled. Here the raw column is an object dtype,
    so it landed in the one-hot list and produced an indicator column per
    distinct date: a model that fits well on a lookup table of dates and can
    match nothing in a later batch. This is the failure that does not announce
    itself, unlike the datetime64 version, which crashes.
    """
    df = retail(rows=400)
    df["date"] = df["date"].astype(str)

    X, _ = _prepare_features(df)

    assert not [c for c in X.columns if c.startswith("date")]
    assert X.shape[1] < 20, f"{X.shape[1]} feature columns from 400 rows"
    assert train_model(df) is not None


def test_a_missing_target_value_does_not_take_down_training():
    """sklearn will not fit on a NaN target, so one missing sales figure used to
    end the upload with `Input y contains NaN`."""
    df = retail()
    df.loc[df.index[:20], "sales"] = None

    assert train_model(df) is not None
    X, y = _prepare_features(df)
    assert len(y) == len(df) - 20
    assert len(X) == len(y)


def test_a_batch_with_no_target_column_trains_nothing():
    assert train_model(retail().drop(columns=["sales"])) is None


def test_evaluating_before_anything_is_trained_returns_nothing():
    assert evaluate_model(retail()) is None


def test_error_rises_when_the_target_moves():
    train_model(retail(seed=1))

    steady = evaluate_model(retail(seed=2))
    shifted = evaluate_model(retail(seed=3, shift=25.0))

    assert shifted["mae"] > steady["mae"] * 5


def test_a_category_the_model_never_saw_does_not_break_scoring():
    train_model(retail(seed=1))

    df = retail(seed=2)
    df.loc[df.index[:50], "product"] = "BRAND_NEW"

    assert evaluate_model(df)["mae"] > 0


def test_mape_excludes_zero_targets_and_says_how_many():
    """Zero targets used to be given a denominator of 1, which reported their
    absolute error as if it were a percentage."""
    train_model(retail(seed=1))

    df = retail(seed=2)
    df.loc[df.index[:40], "sales"] = 0.0
    result = evaluate_model(df)

    assert result["mape_excluded_rows"] == 40
    assert result["mape"] < 1.0


def test_mape_is_absent_when_every_target_is_zero():
    train_model(retail(seed=1))

    df = retail(seed=2)
    df["sales"] = 0.0
    result = evaluate_model(df)

    assert "mape" not in result
    assert result["mape_excluded_rows"] == len(df)
    assert result["mae"] > 0  # the absolute error is still reportable
