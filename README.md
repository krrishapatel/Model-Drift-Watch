# DriftRadar

ML data drift + model health monitoring for retail sales datasets.

## Quickstart (Non-Docker)

1. Create and activate a Python 3.11 venv:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install deps:

```bash
python -m pip install -r requirements.txt
```

3. Generate sample data:

```bash
python scripts/seed_sample.py
```

4. Start the API:

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```

5. Upload a CSV:

```bash
curl -F "file=@sample_retail.csv" http://localhost:8000/ingest
```

6. Open dashboard:

```
http://localhost:8000/
```

## Quickstart (Docker)

1. Start services:

```bash
docker compose up --build
```

2. Generate sample data (inside the container so deps are available):

```bash
docker compose run --rm api python scripts/seed_sample.py
```

3. Upload a CSV:

```bash
curl -F "file=@sample_retail.csv" http://localhost:8000/ingest
```

4. Open dashboard:

```
http://localhost:8000/
```

## API

- `POST /ingest` CSV upload
- `GET /batches`
- `GET /drift?feature=...`
- `GET /quality?batch_id=...`
- `GET /model_health`

## Notes

- Reference window spans the first `REFERENCE_WINDOW_DAYS` days across batches.
- Model health uses a baseline RandomForestRegressor on the reference data.
- Email alerts are optional; set `ALERT_EMAIL_ENABLED=true` and SMTP env vars.

## Email Alerts

Set in `.env` or environment:

```
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_TO=you@example.com
ALERT_EMAIL_FROM=driftradar@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_user
SMTP_PASSWORD=your_password
```
