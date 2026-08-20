"""Reference stats, current stats, and the comparison between them.

A datetime column is measured as epoch seconds and reported as a numeric
feature, so the reference stats for it hold second-scale bin edges. Everything
that turns a column into floats goes through metrics.to_float_series so nothing
here can disagree about that unit.
"""

import numpy as np
import pandas as pd

from .metrics import (
    align_categories,
    categorical_counts,
    chi_square,
    hist_on_bins,
    js_divergence,
    kl_divergence,
    ks_test,
    numeric_hist,
    psi,
    to_float_series,
)


def build_reference_stats(df: pd.DataFrame, schema: dict):
    stats = {}
    for col_info in schema["columns"]:
        col = col_info["name"]
        col_type = col_info["type"]
        if col not in df.columns:
            continue
        series = df[col]
        if col_type in ("numeric", "datetime"):
            # A datetime is stored as epoch seconds under type "numeric", so the
            # comparison path does not need to know which it started as.
            clean = to_float_series(series)
            stats[col] = {
                "type": "numeric",
                "min": float(clean.min()) if not clean.empty else None,
                "max": float(clean.max()) if not clean.empty else None,
                "hist": numeric_hist(series, bins=10),
            }
        elif col_type in ("categorical", "boolean"):
            counts = categorical_counts(series)
            stats[col] = {
                "type": "categorical",
                "categories": counts["categories"],
                "counts": counts["counts"],
            }
    return stats


def build_current_stats(df: pd.DataFrame, reference_stats: dict | None):
    stats = {}
    for col in df.columns:
        series = df[col]
        ref = (reference_stats or {}).get(col)
        if ref and ref["type"] == "numeric" and ref["hist"]["bins"]:
            # Same bin edges as the reference, or the two histograms would not be
            # comparable at all.
            stats[col] = {"type": "numeric", "hist": hist_on_bins(series, ref["hist"]["bins"])}
            continue
        if ref and ref["type"] == "categorical":
            counts = categorical_counts(series)
            stats[col] = {
                "type": "categorical",
                "categories": counts["categories"],
                "counts": counts["counts"],
            }
            continue
        # No reference for this column: it is new, or this is the first batch.
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_datetime64_any_dtype(series):
                stats[col] = {"type": "numeric", "hist": numeric_hist(series, bins=10)}
            else:
                counts = categorical_counts(series)
                stats[col] = {
                    "type": "categorical",
                    "categories": counts["categories"],
                    "counts": counts["counts"],
                }
        else:
            stats[col] = {"type": "numeric", "hist": numeric_hist(series, bins=10)}
    return stats


def compute_drift(
    df: pd.DataFrame,
    reference_stats: dict,
    reference_df: pd.DataFrame | None,
    rolling_stats: dict | None,
):
    metrics = []
    n_rows = len(df)

    current_stats = build_current_stats(df, reference_stats)

    def add(feature, metric, value, comparison):
        metrics.append(
            {"feature": feature, "metric": metric, "value": value, "comparison": comparison}
        )

    for col, ref in reference_stats.items():
        cur = current_stats.get(col)
        if not cur or cur["type"] != ref["type"]:
            continue
        if ref["type"] == "numeric":
            ref_counts = np.array(ref["hist"]["counts"], dtype=float)
            cur_counts = np.array(cur["hist"]["counts"], dtype=float)
            if ref_counts.size == 0 or cur_counts.size == 0:
                continue
            add(col, "psi", psi(ref_counts, cur_counts), "reference")
            add(col, "kl", kl_divergence(ref_counts, cur_counts), "reference")
            if reference_df is not None and col in reference_df.columns:
                statistic, pvalue = ks_test(reference_df[col], df[col])
                add(col, "ks_statistic", statistic, "reference")
                add(col, "ks_pvalue", pvalue, "reference")
        else:
            ref_counts, cur_counts = align_categories(ref, cur)
            add(col, "jsd", js_divergence(ref_counts, cur_counts), "reference")
            statistic, pvalue = chi_square(ref_counts, cur_counts, n_rows)
            add(col, "chi_square", statistic, "reference")
            add(col, "chi_square_pvalue", pvalue, "reference")

    for col, prev in (rolling_stats or {}).items():
        cur = current_stats.get(col)
        if not cur or cur["type"] != prev["type"]:
            continue
        if prev["type"] == "numeric":
            prev_counts = np.array(prev["hist"]["counts"], dtype=float)
            cur_counts = np.array(cur["hist"]["counts"], dtype=float)
            if prev_counts.size == 0 or cur_counts.size == 0:
                continue
            add(col, "psi", psi(prev_counts, cur_counts), "rolling")
        else:
            prev_counts, cur_counts = align_categories(prev, cur)
            add(col, "jsd", js_divergence(prev_counts, cur_counts), "rolling")

    return metrics, current_stats
