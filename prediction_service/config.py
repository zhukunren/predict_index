"""Runtime configuration for the prediction service."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_or_create_session_secret(root: Path) -> str:
    configured = os.getenv("PREDICTION_SERVICE_SESSION_SECRET")
    if configured:
        return configured

    secret_path = root / "session_secret.txt"
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()

    secret = secrets.token_urlsafe(48)
    secret_path.write_text(secret + "\n", encoding="utf-8")
    return secret


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings intentionally keep credentials out of the database and archives."""

    root_dir: Path
    database_url: str
    archive_dir: Path
    local_feature_path: Path
    history_start_date: str
    validation_days: int
    signal_engine: str
    admin_username: str
    admin_password: str | None
    session_secret: str
    cookie_secure: bool
    scheduled_refresh_enabled: bool
    scheduled_refresh_hour: int
    scheduled_refresh_minute: int
    retry_count: int
    retry_sleep_seconds: float
    rate_limit_sleep_seconds: float
    index_global_min_interval: float

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(
            os.getenv("PREDICTION_SERVICE_HOME", PROJECT_ROOT / "service_data")
        ).resolve()
        root.mkdir(parents=True, exist_ok=True)
        archive_dir = root / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        database_url = os.getenv(
            "PREDICTION_SERVICE_DATABASE_URL",
            f"sqlite:///{(root / 'prediction_service.db').as_posix()}",
        )
        return cls(
            root_dir=root,
            database_url=database_url,
            archive_dir=archive_dir,
            local_feature_path=Path(
                os.getenv(
                    "PREDICTION_SERVICE_FEATURE_CSV",
                    PROJECT_ROOT / "market_data" / "merged_features.csv",
                )
            ).resolve(),
            history_start_date=os.getenv(
                "PREDICTION_SERVICE_HISTORY_START_DATE", "20200101"
            ),
            validation_days=int(
                os.getenv("PREDICTION_SERVICE_VALIDATION_DAYS", "60")
            ),
            signal_engine=os.getenv(
                "PREDICTION_SERVICE_SIGNAL_ENGINE", "state_veto_rule"
            ),
            admin_username=os.getenv("PREDICTION_SERVICE_ADMIN_USERNAME", "admin"),
            admin_password=os.getenv("PREDICTION_SERVICE_ADMIN_PASSWORD"),
            session_secret=_load_or_create_session_secret(root),
            cookie_secure=_env_bool("PREDICTION_SERVICE_COOKIE_SECURE", False),
            scheduled_refresh_enabled=_env_bool(
                "PREDICTION_SERVICE_SCHEDULED_REFRESH",
                bool(os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN")),
            ),
            scheduled_refresh_hour=int(
                os.getenv("PREDICTION_SERVICE_REFRESH_HOUR", "18")
            ),
            scheduled_refresh_minute=int(
                os.getenv("PREDICTION_SERVICE_REFRESH_MINUTE", "15")
            ),
            retry_count=int(os.getenv("PREDICTION_SERVICE_RETRIES", "3")),
            retry_sleep_seconds=float(
                os.getenv("PREDICTION_SERVICE_RETRY_SLEEP_SECONDS", "1")
            ),
            rate_limit_sleep_seconds=float(
                os.getenv("PREDICTION_SERVICE_RATE_LIMIT_SLEEP_SECONDS", "65")
            ),
            index_global_min_interval=float(
                os.getenv("PREDICTION_SERVICE_INDEX_GLOBAL_MIN_INTERVAL", "6.2")
            ),
        )

    @classmethod
    def for_test(cls, root_dir: Path, **overrides: object) -> "Settings":
        """Build an isolated configuration without reading test-machine secrets."""

        root = root_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        values: dict[str, object] = {
            "root_dir": root,
            "database_url": f"sqlite:///{(root / 'test.db').as_posix()}",
            "archive_dir": root / "archives",
            "local_feature_path": root / "merged_features.csv",
            "history_start_date": "20200101",
            "validation_days": 60,
            "signal_engine": "state_veto_rule",
            "admin_username": "admin",
            "admin_password": "test-password",
            "session_secret": "test-session-secret-not-for-production",
            "cookie_secure": False,
            "scheduled_refresh_enabled": False,
            "scheduled_refresh_hour": 18,
            "scheduled_refresh_minute": 15,
            "retry_count": 0,
            "retry_sleep_seconds": 0.0,
            "rate_limit_sleep_seconds": 0.0,
            "index_global_min_interval": 0.0,
        }
        values.update(overrides)
        archive_dir = Path(values["archive_dir"])
        archive_dir.mkdir(parents=True, exist_ok=True)
        return cls(**values)  # type: ignore[arg-type]
