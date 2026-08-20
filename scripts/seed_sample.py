"""Write two sample CSVs: a reference set and a drifted batch to compare to it.

There used to be one file, which meant the quickstart could only ever show a
batch compared against itself. It also wrote to the working directory, so where
the file landed depended on where you ran it from. Both files are 800 rows,
above the default MIN_ROWS_FOR_DRIFT_ALERT, so the drifted one actually alerts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROWS = 800


def make(seed, start, sales_shift=0.0, product_weights=(0.34, 0.33, 0.33), missing=0):
    rng = np.random.default_rng(seed)
    product = rng.choice(["A", "B", "C"], size=ROWS, p=product_weights)
    sales = 20 + sales_shift + 5 * (product == "A") + rng.normal(0, 3, size=ROWS)
    df = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=ROWS, freq="D"),
            "product": product,
            "store": rng.choice(["north", "south"], size=ROWS),
            "sales": sales,
        }
    )
    if missing:
        df.loc[df.index[:missing], "sales"] = np.nan
    return df


reference = make(seed=42, start="2024-01-01")
# Sales up by two standard deviations and product A twice as common: enough to
# move PSI, the KS statistic and the chi-square test all at once.
current = make(
    seed=7,
    start="2026-03-11",
    sales_shift=6.0,
    product_weights=(0.6, 0.2, 0.2),
    missing=60,
)

for name, df in (("sample_retail.csv", reference), ("sample_drifted.csv", current)):
    df.to_csv(ROOT / name, index=False)
    print(f"Wrote {ROOT / name} ({len(df)} rows)")
