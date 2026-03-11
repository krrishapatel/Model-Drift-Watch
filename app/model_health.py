import os
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .config import settings


MODEL_PATH = "./data/models/model.joblib"


def _prepare_features(df: pd.DataFrame):
    df = df.copy()
    if settings.time_column in df.columns:
        ts = pd.to_datetime(df[settings.time_column], errors="coerce")
        df["dow"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["day"] = ts.dt.day
    if settings.target_column in df.columns:
        y = df[settings.target_column].astype(float)
        X = df.drop(columns=[settings.target_column])
    else:
        y = None
        X = df

    if settings.category_columns:
        cat_cols = [c.strip() for c in settings.category_columns.split(",") if c.strip()]
    else:
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    X = X.fillna(0)
    return X, y


def train_model(reference_df: pd.DataFrame):
    X, y = _prepare_features(reference_df)
    if y is None or y.empty:
        return None
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump({"model": model, "columns": X.columns.tolist()}, MODEL_PATH)
    return model


def evaluate_model(df: pd.DataFrame):
    if not os.path.exists(MODEL_PATH):
        return None
    payload = load(MODEL_PATH)
    model = payload["model"]
    columns = payload["columns"]

    X, y = _prepare_features(df)
    if y is None or y.empty:
        return None

    # align columns
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    X = X[columns]

    preds = model.predict(X)
    mae = float(mean_absolute_error(y, preds))
    mape = float((abs((y - preds) / y.replace(0, 1))).mean())
    return {"mae": mae, "mape": mape}
