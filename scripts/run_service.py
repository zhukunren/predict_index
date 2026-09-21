"""Run the prediction service locally or behind a reverse proxy."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from filelock import FileLock

import uvicorn

from prediction_service.config import DEFAULT_CONFIG_PATH, Settings
from prediction_service.service import PredictionService
from prediction_service.web import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="运行上证指数预测服务。")
    parser.add_argument(
        "--promote-compatible-release",
        action="store_true",
        help="经历史预测完全一致校验后，创建新的模型发布版本。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="服务配置文件路径，默认使用项目根目录的 config.ini。",
    )
    parser.add_argument("--port", type=int, help="覆盖本次运行的监听端口。")
    parser.add_argument("--no-scheduler", action="store_true", help="本次运行关闭自动刷新。")
    arguments = parser.parse_args(argv)
    settings = Settings.from_config(arguments.config)
    if arguments.port is not None:
        if not 1 <= arguments.port <= 65535:
            parser.error("端口必须在 1 到 65535 之间。")
        settings = replace(settings, port=arguments.port)
    if arguments.no_scheduler:
        settings = replace(settings, scheduled_refresh_enabled=False)
    if arguments.promote_compatible_release:
        with FileLock(str(settings.root_dir / "service.lock"), timeout=0):
            service = PredictionService(settings)
            try:
                service.initialize(bootstrap=False)
                artifact = service.promote_compatible_release(actor="cli")
                if artifact is None:
                    print("当前代码已经是活动模型版本。")
                else:
                    print(f"已发布兼容模型版本：{artifact.release_id}")
                    print(f"新快照：{artifact.snapshot_id}")
            finally:
                service.shutdown()
        raise SystemExit(0)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
