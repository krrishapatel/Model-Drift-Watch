import pandas as pd


def compute_quality(df: pd.DataFrame, reference_stats: dict | None):
    metrics = []
    total_rows = len(df)

    # Duplicate rows
    if total_rows > 0:
        dup_rate = float(df.duplicated().mean())
        metrics.append({"feature": None, "metric": "duplicate_rate", "value": dup_rate})

    for col in df.columns:
        series = df[col]
        missing_rate = float(series.isna().mean()) if total_rows > 0 else 0.0
        metrics.append({"feature": col, "metric": "missing_rate", "value": missing_rate})

        if reference_stats and col in reference_stats:
            ref = reference_stats[col]
            if ref.get("type") == "numeric":
                min_val = ref.get("min")
                max_val = ref.get("max")
                if min_val is not None and max_val is not None and total_rows > 0:
                    out_of_range = series.dropna().astype(float)
                    if not out_of_range.empty:
                        rate = float(((out_of_range < min_val) | (out_of_range > max_val)).mean())
                        metrics.append({"feature": col, "metric": "out_of_range_rate", "value": rate})
            if ref.get("type") == "categorical":
                ref_cats = set(ref.get("categories", []))
                if ref_cats:
                    cur_cats = set(series.dropna().astype(str).unique().tolist())
                    unexpected = len(cur_cats - ref_cats)
                    rate = float(unexpected / max(len(cur_cats), 1))
                    metrics.append({"feature": col, "metric": "unexpected_category_rate", "value": rate})

    return metrics
