import numpy as np
import pandas as pd
from scipy import stats


NUMERIC_TYPES = ("int", "float")


def infer_schema(df: pd.DataFrame) -> dict:
    schema = []
    for col in df.columns:
        series = df[col]
        col_type = "categorical"
        if pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
        elif pd.api.types.is_bool_dtype(series):
            col_type = "boolean"

        schema.append(
            {
                "name": col,
                "type": col_type,
                "nullable": bool(series.isna().any()),
            }
        )
    return {"columns": schema}


def numeric_hist(series: pd.Series, bins: int = 10, min_val=None, max_val=None):
    clean = series.dropna().astype(float)
    if clean.empty:
        return {"bins": [], "counts": []}
    if min_val is None:
        min_val = float(clean.min())
    if max_val is None:
        max_val = float(clean.max())
    if min_val == max_val:
        max_val = min_val + 1.0
    counts, edges = np.histogram(clean, bins=bins, range=(min_val, max_val))
    counts = counts.astype(float)
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return {"bins": edges.tolist(), "counts": counts.tolist()}


def categorical_counts(series: pd.Series, top_n: int = 50):
    clean = series.dropna().astype(str)
    if clean.empty:
        return {"categories": [], "counts": []}
    counts = clean.value_counts().head(top_n)
    total = counts.sum()
    probs = (counts / total).tolist()
    return {"categories": counts.index.tolist(), "counts": probs}


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    expected = np.clip(expected, eps, 1)
    actual = np.clip(actual, eps, 1)
    return float(np.sum((expected - actual) * np.log(expected / actual)))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def ks_test(ref: pd.Series, cur: pd.Series) -> float:
    ref_clean = ref.dropna().astype(float)
    cur_clean = cur.dropna().astype(float)
    if ref_clean.empty or cur_clean.empty:
        return 1.0
    return float(stats.ks_2samp(ref_clean, cur_clean).pvalue)


def chi_square(ref_counts: np.ndarray, cur_counts: np.ndarray, eps: float = 1e-6) -> float:
    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)
    chi, _ = stats.chisquare(f_obs=cur_counts, f_exp=ref_counts)
    return float(chi)


def align_categories(ref_stats: dict, cur_stats: dict):
    categories = list(dict.fromkeys(ref_stats["categories"] + cur_stats["categories"]))
    ref_map = dict(zip(ref_stats["categories"], ref_stats["counts"]))
    cur_map = dict(zip(cur_stats["categories"], cur_stats["counts"]))
    ref = np.array([ref_map.get(c, 0.0) for c in categories], dtype=float)
    cur = np.array([cur_map.get(c, 0.0) for c in categories], dtype=float)
    ref = ref / ref.sum() if ref.sum() > 0 else ref
    cur = cur / cur.sum() if cur.sum() > 0 else cur
    return ref, cur
