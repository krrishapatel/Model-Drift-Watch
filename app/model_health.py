"""A baseline regressor on the reference data, scored on every later batch.

The point of the score is the trend. The first reading is taken on the same rows
the model was fitted on, so it is optimistic by construction; compare batches
after the reference window to each other, not to that first number.
"""

import os

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .config import settings


def _prepare_features(df: pd.DataFrame):
    df = df.copy()
    if settings.time_column in df.columns:
        ts = pd.to_datetime(df[settings.time_column], errors="coerce")
        df["dow"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["day"] = ts.dt.day
        # The raw column has to go once the parts of it worth learning from are
        # out. Left in, it reached sklearn either as a datetime64 column, which
        # raises DTypePromotionError and killed the first upload every time, or,
        # when the batch was reloaded from CSV and the column was still strings,
        # as one indicator column per distinct date. The second is worse than the
        # crash: it fits and it scores well, on a lookup table of dates that
        # cannot match anything in a later batch.
        df = df.drop(columns=[settings.time_column])

    if settings.target_column in df.columns:
        # Rows with no target teach nothing and cannot be scored, and sklearn
        # refuses to fit on a NaN target at all, so a batch with one missing
        # sales figure used to take down the whole upload. Nulls in the features
        # are a different matter: those are filled below.
        df = df[df[settings.target_column].notna()]
        y = df[settings.target_column].astype(float)
        X = df.drop(columns=[settings.target_column])
    else:
        y = None
        X = df

    if settings.category_columns:
        cat_cols = [c.strip() for c in settings.category_columns.split(",") if c.strip()]
        cat_cols = [c for c in cat_cols if c in X.columns]
    else:
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    # Anything still non-numeric would reach sklearn as an object column, and
    # there is no useful default for it.
    X = X.select_dtypes(include=["number", "bool"])
    X = X.fillna(0)
    return X, y


def train_model(reference_df: pd.DataFrame):
    X, y = _prepare_features(reference_df)
    if y is None or y.empty or X.empty or X.shape[1] == 0:
        return None
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(settings.artifact_path), exist_ok=True)
    dump({"model": model, "columns": X.columns.tolist()}, settings.artifact_path)
    return model


def evaluate_model(df: pd.DataFrame):
    if not os.path.exists(settings.artifact_path):
        return None
    payload = load(settings.artifact_path)
    model = payload["model"]
    columns = payload["columns"]

    X, y = _prepare_features(df)
    if y is None or y.empty:
        return None

    # A category the reference never saw becomes a column the model does not
    # know; a category it saw and this batch lacks has to be filled back in.
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    X = X[columns]

    preds = model.predict(X)
    result = {"mae": float(mean_absolute_error(y, preds))}

    # MAPE is undefined where the target is zero. Those rows used to be given a
    # denominator of 1, which quietly reported their absolute error as a
    # percentage. They are excluded and counted instead.
    nonzero = y != 0
    if nonzero.any():
        result["mape"] = float((abs((y[nonzero] - preds[nonzero]) / y[nonzero])).mean())
    result["mape_excluded_rows"] = float((~nonzero).sum())
    return result
