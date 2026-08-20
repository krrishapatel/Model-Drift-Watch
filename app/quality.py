"""Per-batch data quality, independent of any drift metric.

Every rate here is a fraction of rows, so the numbers are comparable to each
other and to a threshold. `missing_rate` is over all rows; `out_of_range_rate`
and `unexpected_category_rate` are over the rows that have a value, since a null
is already counted by `missing_rate` and should not be charged twice.
"""

import pandas as pd

from .metrics import to_float_series


def compute_quality(df: pd.DataFrame, reference_stats: dict | None):
    metrics = []
    total_rows = len(df)

    if total_rows > 0:
        dup_rate = float(df.duplicated().mean())
        metrics.append({"feature": None, "metric": "duplicate_rate", "value": dup_rate})

    for col in df.columns:
        series = df[col]
        missing_rate = float(series.isna().mean()) if total_rows > 0 else 0.0
        metrics.append({"feature": col, "metric": "missing_rate", "value": missing_rate})

        ref = (reference_stats or {}).get(col)
        if not ref or total_rows == 0:
            continue

        if ref.get("type") == "numeric":
            min_val = ref.get("min")
            max_val = ref.get("max")
            if min_val is None or max_val is None:
                continue
            values = to_float_series(series)
            if values.empty:
                continue
            rate = float(((values < min_val) | (values > max_val)).mean())
            metrics.append({"feature": col, "metric": "out_of_range_rate", "value": rate})

        elif ref.get("type") == "categorical":
            ref_cats = set(ref.get("categories", []))
            if not ref_cats:
                continue
            present = series.dropna().astype(str)
            if present.empty:
                continue
            # Rows, not distinct values. Dividing unexpected labels by the number
            # of labels present made one bad row in a thousand read as 0.33 while
            # five hundred bad rows in the same thousand read as 0.50, so the
            # number moved in the wrong direction as the problem got worse.
            rate = float((~present.isin(ref_cats)).mean())
            metrics.append({"feature": col, "metric": "unexpected_category_rate", "value": rate})

    return metrics
