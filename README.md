# DriftRadar

Data drift and model health monitoring for tabular retail sales data. Upload a
CSV, get distribution drift against a reference batch, in-batch quality metrics,
baseline model error, and alerts when something crosses a threshold.

## Quickstart

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/seed_sample.py
uvicorn app.main:app --reload
```

Then open http://localhost:8000/.

`seed_sample.py` writes two files. Upload the reference set first, then the
drifted batch:

```bash
curl -F "file=@sample_retail.csv" "http://localhost:8000/ingest?role=reference"
curl -F "file=@sample_drifted.csv" "http://localhost:8000/ingest?role=current"
```

The second upload should raise PSI, KS and chi-square alerts on `sales` and
`product`, plus a missing-rate alert. They show up on the dashboard and at
`GET /alerts`.

Redis is optional. If it is not reachable, `/ingest` processes the batch inline
in the request instead of queueing it.

## Quickstart (Docker)

```bash
docker compose up --build
docker compose run --rm api python scripts/seed_sample.py
curl -F "file=@sample_retail.csv" "http://localhost:8000/ingest?role=reference"
```

## API

- `POST /ingest?role=auto|reference|current` CSV upload. 400 on an unparseable
  or empty CSV, 413 above `MAX_UPLOAD_MB`.
- `GET /batches`
- `GET /drift?feature=&batch_id=&metric=`
- `GET /quality?feature=&batch_id=&metric=`
- `GET /model_health`
- `GET /alerts?batch_id=&status=`

## Reference sets and the `role` parameter

Drift is only meaningful against something. `role` says what an upload is for:

- `reference` joins the reference set and is what later batches are compared to.
- `current` is scored against the existing reference and never joins it.
- `auto` (the default) joins the reference set if it was uploaded within
  `REFERENCE_WINDOW_DAYS` of the first batch. That is upload time, not the dates
  inside the file, so uploading a backlog in one sitting puts all of it in the
  reference. Use the explicit roles for that case.

The old behaviour had no `role`, so every upload joined the reference set and was
then compared against a reference that already included itself. Drift measured
against yourself is close to zero by construction, which is why a batch could
look clean no matter what was in it.

## Metrics

Numeric features: PSI, KL divergence, and the two-sample KS statistic with its
p-value. Categorical features: Jensen-Shannon distance and a chi-square test.
Quality, per batch: missing rate, out-of-range rate, unexpected category rate,
duplicate row rate. Model: MAE and MAPE from a baseline RandomForestRegressor.

Three things worth knowing about the alerting:

**Small batches are not alerted on.** Null PSI on ten bins runs at roughly
`2(k-1)/n`, so at 100 rows a drift-free comparison has a median PSI near 0.22 and
57% of comparisons exceed the conventional 0.2 threshold. At 300 rows that is 2%,
at 1000 rows it is none. Rather than move the threshold off the industry
convention, batches under `MIN_ROWS_FOR_DRIFT_ALERT` (default 500) get their
metrics computed and stored but not alerted on.

**KS needs both gates.** A KS p-value shrinks with sample size for a shift of any
size, so alerting on the p-value alone fires at the threshold rate on features
with no drift at all. An alert requires the p-value below
`KS_PVALUE_THRESHOLD` and the statistic above `KS_STATISTIC_THRESHOLD`.

**The time column does not raise drift alerts.** A newer batch covers later
dates, so its date histogram shares no bins with the reference and PSI lands near
its maximum on every upload regardless of the data: 12.4 on the two sample files,
which differ only in when they were collected. Same for its out-of-range rate,
which is 1.0 by construction. Those metrics are still computed, stored and
plotted, they just do not alert. A missing date still does.

MAPE is computed only over rows with a nonzero target, and
`mape_excluded_rows` reports how many were left out. Zero targets used to be
given a denominator of 1, which reported absolute error as a percentage.

## Configuration

Every setting is an environment variable or a line in `.env`. Paths resolve
against the repo root, not the working directory.

```
DATABASE_URL=sqlite:///./drift_radar.db
REDIS_URL=redis://redis:6379/0
TIME_COLUMN=date
TARGET_COLUMN=sales
CATEGORY_COLUMNS=product,store
REFERENCE_WINDOW_DAYS=28
MAX_UPLOAD_MB=100
MIN_ROWS_FOR_DRIFT_ALERT=500
PSI_THRESHOLD=0.2
JSD_THRESHOLD=0.1
KS_PVALUE_THRESHOLD=0.05
KS_STATISTIC_THRESHOLD=0.1
CHI_SQUARE_PVALUE_THRESHOLD=0.05
MISSING_RATE_THRESHOLD=0.05
OUT_OF_RANGE_RATE_THRESHOLD=0.01
UNEXPECTED_CATEGORY_RATE_THRESHOLD=0.01
DUPLICATE_RATE_THRESHOLD=0.1
```

Email alerts are off by default:

```
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_TO=you@example.com
ALERT_EMAIL_FROM=driftradar@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_user
SMTP_PASSWORD=your_password
```

## Tests

```bash
python -m pytest
ruff check .
ruff format --check .
```

CI runs all three, plus an import check from a different working directory.

## Upgrading an existing database

The `batches` table gained a `role` column. SQLite will not add it to a database
created before this change, so either delete `drift_radar.db` and re-upload, or:

```sql
ALTER TABLE batches ADD COLUMN role VARCHAR DEFAULT 'auto';
```

## Known gaps

- No schema drift detection. A new or dropped column is skipped rather than
  alerted on.
- The reference set is decided per upload. There is no way to re-designate an
  existing batch without re-uploading it.
- `data/raw` grows without bound. Nothing prunes old uploads.
