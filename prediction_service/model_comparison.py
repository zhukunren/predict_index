"""Administrator comparisons on identical dates with explicit sample provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .archive import sha256_bytes
from .forecast_models import MODELS, MODEL_KEYS, PRODUCTION_KEY
from .metrics import statistics, clean_records


def active_models(service):
    publication, snapshot, release = service._load_active_context()
    service.read_published((publication, snapshot, release))
    manifest = json.loads(Path(snapshot.manifest_path).read_text(encoding="utf-8"))
    models = manifest.get("models", {})
    return snapshot, manifest, models


def model_csv(service, key):
    from .service import PredictionDriftError

    if key not in MODEL_KEYS:
        raise FileNotFoundError(key)
    snapshot, manifest, models = active_models(service)
    if key not in models:
        raise FileNotFoundError(key)
    filename = models[key]["results_file"]
    if Path(filename).name != filename:
        raise PredictionDriftError("模型结果归档路径无效。")
    content = (Path(snapshot.manifest_path).parent / filename).read_bytes()
    if sha256_bytes(content) != manifest["files"].get(filename):
        raise PredictionDriftError("模型 CSV 校验失败。")
    return content


def comparison_data(service, days=60, sample="all"):
    from .service import PredictionDriftError

    if not 1 <= days <= 5000 or sample not in {"all", "live"}:
        raise ValueError("无效的模型对比窗口或样本类型。")
    snapshot, manifest, metadata = active_models(service)
    if set(metadata) != set(MODEL_KEYS):
        return None
    histories = {}
    latest = {}
    for model in MODELS:
        key = model.key
        frame = service.history_frame(metadata[key]["release_id"], snapshot.data_as_of)
        if sample == "live":
            frame = frame.loc[frame["记录来源"].eq("live")]
        histories[key] = frame.set_index("预测目标交易日", drop=False)
        public = pd.read_csv(Path(snapshot.manifest_path).parent / metadata[key]["results_file"],
                             encoding="utf-8-sig", float_precision="round_trip")
        pending = public.loc[public["结果类型"].eq("次日预测")]
        latest[key] = clean_records(pending)[-1] if len(pending) else None
    # Anchor the requested window to production. Missing shadow dates are never
    # replaced with older, potentially easier observations.
    window = histories[PRODUCTION_KEY].tail(days).index.tolist()
    common = [date for date in window if all(date in frame.index for frame in histories.values())]
    production = histories[PRODUCTION_KEY].reindex(common)
    models, rows, chart = [], [], []
    indexed = {}
    for model in MODELS:
        key = model.key
        frame = histories[key].reindex(common).reset_index(drop=True)
        if (frame["信号日期"].tolist() != production["信号日期"].tolist()
                or frame["次日实际涨跌幅"].tolist() != production["次日实际涨跌幅"].tolist()):
            raise PredictionDriftError("模型对比的目标日、信号日或实际结果不一致。")
        metrics = statistics(frame, days)
        error = frame["预测次日涨跌幅"] - frame["次日实际涨跌幅"]
        hit = frame["预测方向"].eq("上涨").eq(frame["次日实际涨跌幅"].gt(0))
        metrics["rmse"] = float(np.sqrt(np.mean(error ** 2))) if len(frame) else None
        metrics["brier"] = float(np.mean((frame["置信度"] - hit.astype(float)) ** 2)) if len(frame) else None
        models.append({"key": key, "name": model.name, **metadata[key], "latest": latest[key],
                       "metrics": metrics, "available_rows": len(histories[key]),
                       "missing_rows": len(window) - len(histories[key].index.intersection(window))})
        indexed[key] = {str(row["预测目标交易日"]): row for row in metrics["records"]}
    running = {key: 0 for key in MODEL_KEYS}
    for index, date in enumerate(common):
        predictions = {key: indexed[key][str(date)] for key in MODEL_KEYS}
        actual = predictions[PRODUCTION_KEY]["次日实际涨跌幅"]
        rows.append({"target_date": date, "signal_date": predictions[PRODUCTION_KEY]["信号日期"],
                     "actual_return": actual, "predictions": predictions,
                     "disagreement": len({row["预测方向"] for row in predictions.values()}) > 1})
        for key in MODEL_KEYS:
            running[key] += int(predictions[key]["hit"])
        chart.append({"date": date, **{key: running[key] / (index + 1) for key in MODEL_KEYS}})
    return {"models": models, "rows": list(reversed(rows)), "chart": chart,
            "days": days, "sample": sample, "paired_rows": len(common),
            "window_rows": len(window), "missing_rows": len(window) - len(common),
            "start_date": common[0] if common else None, "end_date": common[-1] if common else None,
            "data_as_of": snapshot.data_as_of, "snapshot_id": snapshot.id,
            "production_name": metadata[PRODUCTION_KEY]["name"],
            "health": service.health(), "scheduled": service.settings.scheduled_refresh_enabled,
            "schedule_time": f"{service.settings.scheduled_refresh_hour:02d}:{service.settings.scheduled_refresh_minute:02d}",
            "disagreement_rows": sum(row["disagreement"] for row in rows)}


def comparison_frame(data):
    columns = ["目标交易日", "信号日期", "实际涨跌幅"]
    fields = ("预测方向", "预测次日涨跌幅", "预测次日收盘价", "置信度", "hit", "记录来源", "预测生成时间")
    for model in data["models"]:
        columns.extend(f"{model['name']}_{'命中' if field == 'hit' else field}" for field in fields)
    records = []
    for row in reversed(data["rows"]):
        values = [row["target_date"], row["signal_date"], row["actual_return"]]
        for model in data["models"]:
            values.extend(row["predictions"][model["key"]].get(field) for field in fields)
        records.append(values)
    return pd.DataFrame(records, columns=columns)
