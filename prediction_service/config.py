"""Prediction-service configuration loaded from a local INI file."""

from __future__ import annotations

import configparser
import math
import re
import secrets
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.ini"


class ConfigurationError(ValueError):
    """Raised when the local service configuration is invalid."""


_TRUE_VALUES = {"1", "true", "yes", "on", "是", "开启", "启用", "开"}
_FALSE_VALUES = {"0", "false", "no", "off", "否", "关闭", "禁用", "关"}


def _value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    fallback: str,
) -> str:
    if parser.has_option(section, option):
        return parser.get(section, option).strip()
    return fallback


def _optional(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: str, *, section: str, option: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ConfigurationError(
        f"配置项 [{section}] {option} 必须填写 是/否、开启/关闭 或 true/false。"
    )


def _parse_int(value: str, *, section: str, option: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"配置项 [{section}] {option} 必须是整数。"
        ) from exc


def _parse_float(value: str, *, section: str, option: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"配置项 [{section}] {option} 必须是数字。"
        ) from exc


def _resolve_path(value: str, *, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _load_or_create_session_secret(root: Path, configured: str | None) -> str:
    if configured:
        return configured

    secret_path = root / "session_secret.txt"
    if secret_path.exists():
        existing = secret_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    secret = secrets.token_urlsafe(48)
    secret_path.write_text(secret + "\n", encoding="utf-8")
    return secret


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings; credentials never enter the database or archives."""

    root_dir: Path
    database_url: str
    archive_dir: Path
    local_feature_path: Path
    history_start_date: str
    validation_days: int
    signal_engine: str
    bilstm_shadow_enabled: bool
    bilstm_shadow_validation_days: int
    bilstm_shadow_refit_interval: int
    shadow_archive_dir: Path
    admin_username: str
    admin_password: str | None
    session_secret: str
    cookie_secure: bool
    host: str
    port: int
    tushare_token: str | None
    scheduled_refresh_enabled: bool
    scheduled_refresh_hour: int
    scheduled_refresh_minute: int
    retry_count: int
    retry_sleep_seconds: float
    rate_limit_sleep_seconds: float
    index_global_min_interval: float

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "Settings":
        """Read a UTF-8 ``config.ini`` whose relative paths use its directory."""

        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(
                f"找不到配置文件：{config_file}。请参考 config.ini.example 创建 config.ini。"
            )

        parser = configparser.ConfigParser(
            interpolation=None,
            empty_lines_in_values=False,
            strict=True,
        )
        parser.optionxform = str
        try:
            with config_file.open(encoding="utf-8") as handle:
                parser.read_file(handle)
        except (OSError, configparser.Error) as exc:
            raise ConfigurationError(f"无法读取配置文件 {config_file}：{exc}") from exc

        config_dir = config_file.parent
        root = _resolve_path(
            _value(parser, "服务", "数据目录", "service_data"),
            config_dir=config_dir,
        )
        root.mkdir(parents=True, exist_ok=True)
        archive_dir = root / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        shadow_archive_dir = root / "shadow_archives"
        shadow_archive_dir.mkdir(parents=True, exist_ok=True)

        database_url = _optional(_value(parser, "服务", "数据库URL", ""))
        if database_url is None:
            database_url = f"sqlite:///{(root / 'prediction_service.db').as_posix()}"

        history_start_date = _value(parser, "服务", "历史起始日期", "20200101")
        if not re.fullmatch(r"\d{8}", history_start_date):
            raise ConfigurationError("配置项 [服务] 历史起始日期必须是 YYYYMMDD 格式。")

        validation_days = _parse_int(
            _value(parser, "服务", "循环验证天数", "60"),
            section="服务",
            option="循环验证天数",
        )
        if validation_days < 0:
            raise ConfigurationError("配置项 [服务] 循环验证天数不能小于 0。")

        signal_engine = _value(parser, "服务", "正式预测引擎", "state_veto_rule")
        if not signal_engine:
            raise ConfigurationError("配置项 [服务] 正式预测引擎不能为空。")

        host = _value(parser, "服务", "监听地址", "127.0.0.1")
        if not host:
            raise ConfigurationError("配置项 [服务] 监听地址不能为空。")
        port = _parse_int(
            _value(parser, "服务", "监听端口", "8000"),
            section="服务",
            option="监听端口",
        )
        if not 1 <= port <= 65535:
            raise ConfigurationError("配置项 [服务] 监听端口必须在 1 到 65535 之间。")

        cookie_secure = _parse_bool(
            _value(parser, "服务", "Cookie仅HTTPS", "否"),
            section="服务",
            option="Cookie仅HTTPS",
        )
        tushare_token = _optional(_value(parser, "Tushare", "令牌", ""))
        scheduled_refresh_enabled = _parse_bool(
            _value(
                parser,
                "服务",
                "启用定时刷新",
                "是" if tushare_token else "否",
            ),
            section="服务",
            option="启用定时刷新",
        )
        if scheduled_refresh_enabled and tushare_token is None:
            raise ConfigurationError(
                "启用定时刷新前，必须填写 [Tushare] 令牌。"
            )
        scheduled_refresh_hour = _parse_int(
            _value(parser, "服务", "刷新小时", "18"),
            section="服务",
            option="刷新小时",
        )
        scheduled_refresh_minute = _parse_int(
            _value(parser, "服务", "刷新分钟", "15"),
            section="服务",
            option="刷新分钟",
        )
        if not 0 <= scheduled_refresh_hour <= 23:
            raise ConfigurationError("配置项 [服务] 刷新小时必须在 0 到 23 之间。")
        if not 0 <= scheduled_refresh_minute <= 59:
            raise ConfigurationError("配置项 [服务] 刷新分钟必须在 0 到 59 之间。")

        bilstm_shadow_enabled = _parse_bool(
            _value(parser, "BiLSTM影子", "启用", "否"),
            section="BiLSTM影子",
            option="启用",
        )
        bilstm_shadow_validation_days = _parse_int(
            _value(
                parser,
                "BiLSTM影子",
                "循环验证天数",
                str(validation_days),
            ),
            section="BiLSTM影子",
            option="循环验证天数",
        )
        if bilstm_shadow_validation_days < 0:
            raise ConfigurationError("配置项 [BiLSTM影子] 循环验证天数不能小于 0。")
        bilstm_shadow_refit_interval = _parse_int(
            _value(parser, "BiLSTM影子", "重训间隔交易日", "5"),
            section="BiLSTM影子",
            option="重训间隔交易日",
        )
        if bilstm_shadow_refit_interval < 1:
            raise ConfigurationError("配置项 [BiLSTM影子] 重训间隔交易日必须至少为 1。")

        retry_count = _parse_int(
            _value(parser, "Tushare", "重试次数", "3"),
            section="Tushare",
            option="重试次数",
        )
        retry_sleep_seconds = _parse_float(
            _value(parser, "Tushare", "普通重试等待秒数", "1"),
            section="Tushare",
            option="普通重试等待秒数",
        )
        rate_limit_sleep_seconds = _parse_float(
            _value(parser, "Tushare", "限流等待秒数", "65"),
            section="Tushare",
            option="限流等待秒数",
        )
        index_global_min_interval = _parse_float(
            _value(parser, "Tushare", "港股接口最小间隔秒数", "6.2"),
            section="Tushare",
            option="港股接口最小间隔秒数",
        )
        if retry_count < 0:
            raise ConfigurationError("配置项 [Tushare] 重试次数不能小于 0。")
        timing_values = (
            retry_sleep_seconds,
            rate_limit_sleep_seconds,
            index_global_min_interval,
        )
        if not all(math.isfinite(value) and value >= 0 for value in timing_values):
            raise ConfigurationError("Tushare 等待时间和接口间隔必须是非负有限数字。")

        admin_username = _value(parser, "管理员", "账号", "admin")
        if not admin_username:
            raise ConfigurationError("配置项 [管理员] 账号不能为空。")

        return cls(
            root_dir=root,
            database_url=database_url,
            archive_dir=archive_dir,
            local_feature_path=_resolve_path(
                _value(
                    parser,
                    "服务",
                    "本地特征文件",
                    "market_data/merged_features.csv",
                ),
                config_dir=config_dir,
            ),
            history_start_date=history_start_date,
            validation_days=validation_days,
            signal_engine=signal_engine,
            bilstm_shadow_enabled=bilstm_shadow_enabled,
            bilstm_shadow_validation_days=bilstm_shadow_validation_days,
            bilstm_shadow_refit_interval=bilstm_shadow_refit_interval,
            shadow_archive_dir=shadow_archive_dir,
            admin_username=admin_username,
            admin_password=_optional(_value(parser, "管理员", "密码", "")),
            session_secret=_load_or_create_session_secret(
                root,
                _optional(_value(parser, "服务", "会话密钥", "")),
            ),
            cookie_secure=cookie_secure,
            host=host,
            port=port,
            tushare_token=tushare_token,
            scheduled_refresh_enabled=scheduled_refresh_enabled,
            scheduled_refresh_hour=scheduled_refresh_hour,
            scheduled_refresh_minute=scheduled_refresh_minute,
            retry_count=retry_count,
            retry_sleep_seconds=retry_sleep_seconds,
            rate_limit_sleep_seconds=rate_limit_sleep_seconds,
            index_global_min_interval=index_global_min_interval,
        )

    @classmethod
    def for_test(cls, root_dir: Path, **overrides: object) -> "Settings":
        """Build an isolated configuration without loading local credentials."""

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
            "bilstm_shadow_enabled": False,
            "bilstm_shadow_validation_days": 60,
            "bilstm_shadow_refit_interval": 5,
            "shadow_archive_dir": root / "shadow_archives",
            "admin_username": "admin",
            "admin_password": "test-password",
            "session_secret": "test-session-secret-not-for-production",
            "cookie_secure": False,
            "host": "127.0.0.1",
            "port": 8000,
            "tushare_token": "test-tushare-token",
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
        shadow_archive_dir = Path(values["shadow_archive_dir"])
        shadow_archive_dir.mkdir(parents=True, exist_ok=True)
        return cls(**values)  # type: ignore[arg-type]
