import pandas as pd
import numpy as np

np.random.seed(42)

rows = 500

dates = pd.date_range("2024-01-01", periods=rows, freq="D")
product = np.random.choice(["A", "B", "C"], size=rows)
store = np.random.choice(["north", "south"], size=rows)

sales = 20 + 5 * (product == "A").astype(int) + np.random.normal(0, 3, size=rows)

pd.DataFrame({"date": dates, "product": product, "store": store, "sales": sales}).to_csv("sample_retail.csv", index=False)
print("Wrote sample_retail.csv")
