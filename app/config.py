from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./drift_radar.db"
    redis_url: str = "redis://redis:6379/0"
    data_dir: str = "./data/raw"

    reference_window_days: int = 28
    time_column: str = "date"
    target_column: str = "sales"
    category_columns: str = ""

    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05
    missing_rate_threshold: float = 0.05

    alert_email_enabled: bool = False
    alert_email_to: str = ""
    alert_email_from: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
