from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prediction_service.metrics import statistics
from prediction_service.web import create_app
from test_prediction_service import _features, _service


def _predictions():
    dates = pd.bdate_range("2026-08-18", periods=22).strftime("%Y%m%d").astype(int)
    labels = ["上涨"] * 15 + ["下跌", "上涨", "上涨", "下跌", "上涨", "上涨"]
    return pd.DataFrame({
        "信号日期": dates[:-1],
        "预测目标交易日": dates[1:],
        "预测方向": labels,
        "预测次日涨跌幅": [0.01 if label == "上涨" else -0.01 for label in labels],
        "次日实际涨跌幅": [0.01] * 10 + [-0.01] * 10 + [None],
        "置信度": [0.6] * 21,
    })


def test_concentration_and_streak_expose_missed_down_days():
    frame = _predictions()
    result = statistics(frame)
    direction = result["direction"]
    assert result["rows"] == 20
    assert result["accuracy"] == pytest.approx(0.6)
    assert result["accuracy_lift"] == pytest.approx(0.1)
    assert result["up_recall"] == 1
    assert result["down_recall"] == pytest.approx(0.2)
    assert direction["predicted_up_rate"] == pytest.approx(0.9)
    assert direction["actual_up_rate"] == pytest.approx(0.5)
    assert direction["longest_up"] == {
        "direction": "上涨", "rows": 15,
        "start_date": 20260819, "end_date": 20260908,
        "correct_rows": 10, "accuracy": pytest.approx(10 / 15),
    }
    assert direction["switch_rows"] == 4
    assert direction["switch_rate"] == pytest.approx(4 / 19)
    assert {alert["code"] for alert in direction["alerts"]} == {"direction_concentration", "long_direction_streak"}
    assert all("_sequence" not in row for row in result["records"])


def test_custom_window_truncates_streak_and_ignores_pending():
    result = statistics(_predictions().iloc[::-1], 5)
    assert result["rows"] == 5 and result["available_rows"] == 20
    assert result["direction"]["longest_up"]["rows"] == 2
    assert result["direction"]["longest_up"]["correct_rows"] == 0
    assert result["direction"]["longest_down"]["rows"] == 1
    assert result["direction"]["alerts"] == []
    assert result["accuracy"] == pytest.approx(0.4)
    assert result["date_basis"] == "target_trade_date"


@pytest.mark.parametrize("label, actual", [("上涨", 0.01), ("下跌", -0.01)])
def test_concentration_requires_divergence_from_actual_outcomes(label, actual):
    frame = _predictions().iloc[:20].copy()
    frame["预测方向"] = label
    frame["次日实际涨跌幅"] = actual
    result = statistics(frame)
    assert [alert["code"] for alert in result["direction"]["alerts"]] == ["long_direction_streak"]
    frame.loc[:9, "次日实际涨跌幅"] = -actual
    alerts = statistics(frame)["direction"]["alerts"]
    assert alerts[0]["code"] == "direction_concentration" and alerts[0]["direction"] == label


@pytest.mark.parametrize("gap", ["missing_outcome", "missing_row"])
def test_streaks_do_not_bridge_unsettled_or_missing_trading_days(gap):
    frame = _predictions().iloc[:6].copy()
    if gap == "missing_outcome":
        frame.loc[2, "次日实际涨跌幅"] = None
    else:
        frame = frame.drop(index=2)
    result = statistics(frame)
    assert result["direction"]["longest_up"]["rows"] == 3
    assert result["direction"]["transition_rows"] == 3


def test_empty_single_row_and_legacy_dates_have_defined_statistics():
    frame = _predictions().drop(columns="预测目标交易日")
    for sample in [frame.iloc[:0], frame.iloc[-1:], frame.iloc[:1]]:
        result = statistics(sample)
        json.dumps(result, allow_nan=False)
        assert result["direction"]["switch_rate"] is None
        assert result["direction"]["alerts"] == []
        assert result["date_basis"] == "signal_date"
    assert statistics(frame.iloc[:1])["direction"]["longest_up"]["start_date"] == 20260818


def test_public_metrics_are_read_only_and_use_the_requested_window(tmp_path, monkeypatch):
    service = _service(tmp_path)
    for size in range(8, 12):
        artifact = service.publish_from_features(_features(size), source="test", raw_frames=None, actor="test")
    monkeypatch.setattr(service, "calculation_function", lambda *args: pytest.fail("metrics ran the model"))
    with TestClient(create_app(service.settings, service=service, bootstrap=False)) as client:
        before = client.get("/api/v1/sh000001/latest.csv")
        response = client.get("/api/v1/sh000001/metrics?days=1")
        assert response.status_code == 200
        payload = response.json()
        assert payload["snapshot_id"] == artifact.snapshot_id
        assert payload["metrics"]["rows"] == 1
        assert payload["metrics"]["requested_days"] == 1
        assert payload["recent_20"]["rows"] == 5
        assert "records" not in payload["metrics"]
        assert client.get("/api/v1/sh000001/metrics?days=0").status_code == 422
        assert client.get("/api/v1/sh000001/metrics?days=5001").status_code == 422
        after = client.get("/api/v1/sh000001/latest.csv")
        assert after.content == before.content == artifact.csv_bytes
        assert after.headers["etag"] == before.headers["etag"]
        artifact.archive.results_path.write_bytes(b"corrupt")
        assert client.get("/api/v1/sh000001/metrics").status_code == 503
