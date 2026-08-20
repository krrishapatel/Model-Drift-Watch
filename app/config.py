"""Settings, and the paths everything else resolves against.

The defaults used to be relative ("./data/raw", "./frontend"), so importing
app.main from anywhere but the repo root raised RuntimeError on the static
mount, and a worker started from a different directory wrote its uploads and its
model to a different place than the API read them from. They are anchored to the
repo root here instead. Every one is still overridable by environment variable.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    database_url: str = f"sqlite:///{BASE_DIR / 'drift_radar.db'}"
    redis_url: str = "redis://redis:6379/0"
    data_dir: str = str(BASE_DIR / "data" / "raw")
    # Not "model_path": pydantic reserves the model_ prefix on a settings class.
    artifact_path: str = str(BASE_DIR / "data" / "models" / "model.joblib")
    frontend_dir: str = str(BASE_DIR / "frontend")

    reference_window_days: int = 28
    time_column: str = "date"
    target_column: str = "sales"
    category_columns: str = ""

    max_upload_mb: int = 100

    # Below this many rows, drift metrics are computed and stored but not
    # alerted on. See the note in ingestion.create_alerts for the measurements
    # this default comes from.
    min_rows_for_drift_alert: int = 500

    # Distribution shift, against the reference batch.
    psi_threshold: float = 0.2
    jsd_threshold: float = 0.1
    # A test's p-value says "this shift is not chance", which with a few thousand
    # rows is true of shifts too small to act on. Both gates have to trip: the
    # p-value below its threshold and the effect above its own.
    ks_pvalue_threshold: float = 0.05
    ks_statistic_threshold: float = 0.1
    chi_square_pvalue_threshold: float = 0.05

    # Data quality, within the batch.
    missing_rate_threshold: float = 0.05
    out_of_range_rate_threshold: float = 0.01
    unexpected_category_rate_threshold: float = 0.01
    duplicate_rate_threshold: float = 0.1

    alert_email_enabled: bool = False
    alert_email_to: str = ""
    alert_email_from: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""


settings = Settings()
