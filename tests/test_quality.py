import pandas as pd
from drift_radar.app.quality import compute_quality


def test_missing_rate_and_duplicates():
    df = pd.DataFrame({"a": [1, None, 1], "b": ["x", "x", "x"]})
    metrics = compute_quality(df, reference_stats=None)
    missing = [m for m in metrics if m["metric"] == "missing_rate" and m["feature"] == "a"][0]
    assert missing["value"] == 1 / 3

    dup = [m for m in metrics if m["metric"] == "duplicate_rate"][0]
    assert dup["value"] > 0
