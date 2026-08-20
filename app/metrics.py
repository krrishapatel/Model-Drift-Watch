"""The statistics. Everything here is pure: pandas in, numbers out.

Two notes on reading the numbers this module produces.

Histogram metrics (psi, kl, jsd) run on proportions, so they do not depend on
how many rows a batch has. Test metrics (chi_square, ks) do depend on it, which
is the point of a test, and both are reported with a p-value.

Every empty bin is floored at `eps` before a log is taken. That means a category
or range that is present in the reference and absent now produces a large number
rather than an infinite one. It is still a large number: treat "PSI = 12" as
"this bin emptied out", not as a magnitude worth comparing to another 12.
"""

import numpy as np
import pandas as pd
from scipy import stats


def infer_schema(df: pd.DataFrame) -> dict:
    schema = []
    for col in df.columns:
        series = df[col]
        # Booleans are checked before numerics on purpose. is_numeric_dtype is
        # True for a bool column, so with the other order every boolean column
        # was typed "numeric" and this branch never ran.
        if pd.api.types.is_bool_dtype(series):
            col_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
        else:
            col_type = "categorical"

        schema.append(
            {
                "name": col,
                "type": col_type,
                "nullable": bool(series.isna().any()),
            }
        )
    return {"columns": schema}


def to_float_series(series: pd.Series) -> pd.Series:
    """Nulls dropped, values as floats. Datetimes become epoch seconds.

    Every numeric path used to call `series.dropna().astype(float)` directly,
    which raises TypeError on a datetime column: pandas will not cast a
    DatetimeArray to float. Since a datetime column is stored with type
    "numeric" in the reference stats, that call was reached on the very first
    upload, from four separate places. Converting in one place is the only way
    the four cannot disagree about the unit, which is seconds.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dropna().astype("int64") / 1e9
    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def numeric_hist(series: pd.Series, bins: int = 10, min_val=None, max_val=None):
    clean = to_float_series(series)
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


def hist_on_bins(series: pd.Series, bins: list) -> dict:
    """Bin a batch against the reference edges, keeping the out-of-range rows.

    np.histogram drops anything outside the outermost edges. That silently threw
    away exactly the rows that had drifted, and then the survivors were
    renormalised to sum to 1, so a batch whose shape was unchanged in the middle
    read as no drift no matter how much mass had left the reference range.
    Measured on a normal(20, 3) reference with a quarter of the rows moved out to
    200: PSI came back 0.0165 against a threshold of 0.2.

    Clipping instead of dropping puts that mass in the first or last bin, which
    is what PSI is normally computed on. The distribution inside the reference
    range is unaffected.
    """
    clean = to_float_series(series)
    edges = np.asarray(bins, dtype=float)
    if clean.empty or edges.size < 2:
        return {"bins": list(bins), "counts": []}
    clipped = clean.clip(lower=edges[0], upper=edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    counts = counts.astype(float)
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return {"bins": list(bins), "counts": counts.tolist()}


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


def ks_test(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    """The two-sample KS statistic and its p-value, in that order.

    The statistic is the one to threshold on. It is the largest gap between the
    two cumulative distributions, so it stays put as the batch grows. The
    p-value only answers "could this gap be chance", and with a few thousand
    rows the answer is no for gaps far too small to care about, so alerting on
    the p-value alone fires on drift-free features at whatever rate the
    threshold is set to. Both are reported; the alert wants both.
    """
    ref_clean = to_float_series(ref)
    cur_clean = to_float_series(cur)
    if ref_clean.empty or cur_clean.empty:
        return 0.0, 1.0
    result = stats.ks_2samp(ref_clean, cur_clean)
    return float(result.statistic), float(result.pvalue)


def chi_square(
    ref_props: np.ndarray,
    cur_props: np.ndarray,
    n_rows: int,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Pearson's chi-square over `n_rows` observations, and its p-value.

    Both inputs are proportions. Handing proportions straight to
    scipy.stats.chisquare divides the statistic by the sample size, which
    cancels the only thing the test measures: a 70/30 split against a 50/50
    reference returned 0.16 on ten rows and 0.16 on a hundred thousand. Scaling
    the observed proportions back up to counts restores that. The p-value is
    what the alert thresholds on, since the statistic's scale depends on how
    many categories there are.
    """
    if n_rows <= 0 or ref_props.size == 0:
        return 0.0, 1.0
    expected = np.clip(ref_props, eps, None) * n_rows
    observed = np.clip(cur_props, eps, None) * n_rows
    # chisquare requires both to sum to the same total, and the eps floors above
    # nudge them apart by a fraction of a row.
    expected = expected * (observed.sum() / expected.sum())
    chi, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(chi), float(pvalue)


def align_categories(ref_stats: dict, cur_stats: dict):
    categories = list(dict.fromkeys(ref_stats["categories"] + cur_stats["categories"]))
    ref_map = dict(zip(ref_stats["categories"], ref_stats["counts"]))
    cur_map = dict(zip(cur_stats["categories"], cur_stats["counts"]))
    ref = np.array([ref_map.get(c, 0.0) for c in categories], dtype=float)
    cur = np.array([cur_map.get(c, 0.0) for c in categories], dtype=float)
    ref = ref / ref.sum() if ref.sum() > 0 else ref
    cur = cur / cur.sum() if cur.sum() > 0 else cur
    return ref, cur
