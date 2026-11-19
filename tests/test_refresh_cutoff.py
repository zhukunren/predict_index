from datetime import datetime, timezone
import importlib

import pandas as pd
import pytest

from prediction_service.config import Settings
from prediction_service.service import PredictionService
from test_prediction_service import _fake_calculation, _features


@pytest.mark.parametrize("now,expected", [(datetime(2026, 11, 19, 6, tzinfo=timezone.utc), "20261118"),
    (datetime(2026, 11, 19, 10, 29, tzinfo=timezone.utc), "20261118"),
    (datetime(2026, 11, 19, 10, 30, tzinfo=timezone.utc), "20261119")])
def test_fetch_respects_calendar_publication_cutoff(tmp_path, monkeypatch, now, expected):
    service = PredictionService(Settings.for_test(tmp_path, tushare_token="test-token"), calculation_function=_fake_calculation)
    service.database.initialize()
    dates = pd.date_range("2026-10-01", "2026-11-19")
    service.calendar.store(pd.DataFrame({"cal_date": dates.strftime("%Y%m%d"), "is_open": (dates.weekday < 5).astype(int)}))
    monkeypatch.setattr("prediction_service.service.utcnow", lambda: now)
    captured = []
    def fetch(_start, end, **_kwargs):
        captured.append(end)
        return {"merged_features": _features()}
    monkeypatch.setattr(importlib.import_module("数据拉取脚本_tushare"), "fetch_all", fetch)
    try:
        service.fetch_tushare_features()
        assert captured == [expected]
    finally:
        service.shutdown()
