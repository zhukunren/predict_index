"""Run the prediction service locally or behind a reverse proxy."""

from __future__ import annotations

import os
import argparse

import uvicorn

from prediction_service.config import Settings
from prediction_service.service import PredictionService
from prediction_service.web import create_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行上证指数预测服务。")
    parser.add_argument(
        "--promote-compatible-release",
        action="store_true",
        help="经历史预测完全一致校验后，创建新的模型发布版本。",
    )
    arguments = parser.parse_args()
    if arguments.promote_compatible_release:
        service = PredictionService(Settings.from_env())
        service.initialize(bootstrap=False)
        artifact = service.promote_compatible_release(actor="cli")
        if artifact is None:
            print("当前代码已经是活动模型版本。")
        else:
            print(f"已发布兼容模型版本：{artifact.release_id}")
            print(f"新快照：{artifact.snapshot_id}")
        service.shutdown()
        raise SystemExit(0)
    uvicorn.run(
        create_app(),
        host=os.getenv("PREDICTION_SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("PREDICTION_SERVICE_PORT", "8000")),
        proxy_headers=True,
    )
