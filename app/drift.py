import numpy as np
import pandas as pd
from .metrics import numeric_hist, categorical_counts, psi, kl_divergence, js_divergence, ks_test, chi_square, align_categories


def build_reference_stats(df: pd.DataFrame, schema: dict):
    stats = {}
    for col_info in schema["columns"]:
        col = col_info["name"]
        col_type = col_info["type"]
        series = df[col]
        if col_type == "numeric":
            hist = numeric_hist(series, bins=10)
            stats[col] = {
                "type": "numeric",
                "min": float(series.dropna().astype(float).min()) if not series.dropna().empty else None,
                "max": float(series.dropna().astype(float).max()) if not series.dropna().empty else None,
                "hist": hist,
            }
        elif col_type in ("categorical", "boolean"):
            counts = categorical_counts(series)
            stats[col] = {
                "type": "categorical",
                "categories": counts["categories"],
                "counts": counts["counts"],
            }
        elif col_type == "datetime":
            # treat as numeric timestamp for drift
            ts = pd.to_datetime(series, errors="coerce").dropna().astype("int64") / 1e9
            hist = numeric_hist(pd.Series(ts), bins=10)
            stats[col] = {
                "type": "numeric",
                "min": float(ts.min()) if len(ts) else None,
                "max": float(ts.max()) if len(ts) else None,
                "hist": hist,
            }
    return stats


def build_current_stats(df: pd.DataFrame, reference_stats: dict | None):
    stats = {}
    for col in df.columns:
        series = df[col]
        if reference_stats and col in reference_stats and reference_stats[col]["type"] == "numeric":
            ref_hist = reference_stats[col]["hist"]
            if ref_hist["bins"]:
                bins = np.array(ref_hist["bins"])
                clean = series.dropna().astype(float)
                if clean.empty:
                    stats[col] = {"type": "numeric", "hist": {"bins": ref_hist["bins"], "counts": []}}
                else:
                    counts, _ = np.histogram(clean, bins=bins)
                    counts = counts.astype(float)
                    total = counts.sum()
                    counts = counts / total if total > 0 else counts
                    stats[col] = {"type": "numeric", "hist": {"bins": ref_hist["bins"], "counts": counts.tolist()}}
                continue
        if pd.api.types.is_numeric_dtype(series):
            hist = numeric_hist(series, bins=10)
            stats[col] = {"type": "numeric", "hist": hist}
        else:
            counts = categorical_counts(series)
            stats[col] = {"type": "categorical", "categories": counts["categories"], "counts": counts["counts"]}
    return stats


def compute_drift(df: pd.DataFrame, reference_stats: dict, reference_df: pd.DataFrame | None, rolling_stats: dict | None):
    metrics = []

    current_stats = build_current_stats(df, reference_stats)

    for col, ref in reference_stats.items():
        cur = current_stats.get(col)
        if not cur:
            continue
        if ref["type"] == "numeric" and cur["type"] == "numeric":
            ref_counts = np.array(ref["hist"]["counts"], dtype=float)
            cur_counts = np.array(cur["hist"]["counts"], dtype=float)
            if ref_counts.size == 0 or cur_counts.size == 0:
                continue
            metrics.append({"feature": col, "metric": "psi", "value": psi(ref_counts, cur_counts), "comparison": "reference"})
            metrics.append({"feature": col, "metric": "kl", "value": kl_divergence(ref_counts, cur_counts), "comparison": "reference"})
            if reference_df is not None and col in reference_df.columns:
                metrics.append({"feature": col, "metric": "ks_pvalue", "value": ks_test(reference_df[col], df[col]), "comparison": "reference"})
        if ref["type"] == "categorical" and cur["type"] == "categorical":
            ref_counts, cur_counts = align_categories(ref, cur)
            metrics.append({"feature": col, "metric": "jsd", "value": js_divergence(ref_counts, cur_counts), "comparison": "reference"})
            metrics.append({"feature": col, "metric": "chi_square", "value": chi_square(ref_counts, cur_counts), "comparison": "reference"})

    if rolling_stats:
        for col, prev in rolling_stats.items():
            cur = current_stats.get(col)
            if not cur:
                continue
            if prev["type"] == "numeric" and cur["type"] == "numeric":
                ref_counts = np.array(prev["hist"]["counts"], dtype=float)
                cur_counts = np.array(cur["hist"]["counts"], dtype=float)
                if ref_counts.size == 0 or cur_counts.size == 0:
                    continue
                metrics.append({"feature": col, "metric": "psi", "value": psi(ref_counts, cur_counts), "comparison": "rolling"})
            if prev["type"] == "categorical" and cur["type"] == "categorical":
                ref_counts, cur_counts = align_categories(prev, cur)
                metrics.append({"feature": col, "metric": "jsd", "value": js_divergence(ref_counts, cur_counts), "comparison": "rolling"})

    return metrics, current_stats
