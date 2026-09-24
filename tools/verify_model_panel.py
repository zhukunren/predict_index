"""Browser acceptance checks for the administrator's four-model comparison."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import io
import json
from pathlib import Path
import re

import pandas as pd
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.ini"))
    args = parser.parse_args()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], {}
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1040})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        public = context.request.get(args.url + "/api/v1/sh000001/latest.json")
        assert public.status == 200
        public_payload = public.json()
        assert public_payload["code"] == 0 and public_payload["status"] == 200
        assert public_payload["data"]["msg"] == "success"
        public_rows = public_payload["data"]["items"]
        public_fields = [
            "signal_date", "predicted_next_day_return", "predicted_direction",
            "predicted_next_day_close", "confidence", "actual_next_day_return",
            "direction_prediction_correct",
        ]
        assert len(public_rows) == 61
        assert all(list(row) == public_fields for row in public_rows)
        assert all(row["predicted_direction"] in {"up", "down"} for row in public_rows)
        page.goto(args.url + "/admin/models")
        assert "/admin/login" in page.url
        page.get_by_label("账号", exact=True).fill(config.get("管理员", "账号"))
        page.get_by_label("密码", exact=True).fill(config.get("管理员", "密码"))
        page.get_by_role("button", name="登录", exact=True).click()
        page.wait_for_url(re.compile(r"/admin/$"))
        page.get_by_role("link", name="模型对比", exact=True).click()
        page.wait_for_function("window.Chart && Chart.getChart(document.getElementById('model-comparison-chart'))")
        assert page.locator(".model-card").count() == 4
        assert "期权＋资金流组合" in page.locator(".production-card").text_content()
        assert page.locator(".badge.shadow").count() == 3
        for width, height, label in ((1440, 1040, "desktop"), (390, 844, "mobile"), (320, 800, "small-mobile")):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(200)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"{label} overflow"
            assert page.locator("canvas").is_visible()
            page.screenshot(path=args.output / f"models-{label}.png", full_page=True)
            checks[label] = "layout and chart passed"
        page.set_viewport_size({"width": 1440, "height": 1040})
        page.get_by_label("自定义统计交易日数", exact=True).fill("17")
        page.get_by_role("button", name="应用", exact=True).click()
        page.wait_for_url(re.compile(r"days=17"))
        assert len(json.loads(page.locator("#model-comparison-data").text_content())["chart"]) == 17
        with page.expect_download() as download:
            page.get_by_role("link", name="导出对比 CSV", exact=True).click()
        frame = pd.read_csv(download.value.path(), encoding="utf-8-sig")
        assert len(frame) == 17 and len(frame.columns) == 31
        page.get_by_label("仅看方向分歧", exact=True).check()
        assert all(value == "true" for value in page.locator(".daily-model-table tbody tr:visible").evaluate_all("rows => rows.map(row => row.dataset.disagreement)"))
        page.get_by_label("样本范围", exact=True).select_option("live")
        page.get_by_role("button", name="应用", exact=True).click()
        page.wait_for_url(re.compile(r"sample=live"))
        assert "事前预测尚未形成共同结算结果" in page.text_content("main")
        page.screenshot(path=args.output / "models-prospective-empty.png", full_page=True)
        page.goto(args.url + "/admin/models?days=60")
        page.get_by_role("button", name="下一页", exact=True).click()
        assert page.locator("[data-comparison-page]").text_content().startswith("2 /")
        data = context.request.get(args.url + "/admin/api/models?days=60").json()
        assert data["paired_rows"] == 60
        assert {model["metrics"]["rows"] for model in data["models"]} == {60}
        assert public.headers["x-model-release"] == data["models"][0]["release_id"]
        for model in data["models"]:
            csv = context.request.get(args.url + "/admin/models/" + model["key"] + "/latest.csv")
            assert csv.status == 200
            rows = pd.read_csv(io.BytesIO(csv.body()), encoding="utf-8-sig")
            assert len(rows) == 61 and set(rows["模型版本"]) == {model["release_id"]}
        page.goto(args.url + "/admin/")
        assert "期权＋资金流组合" in page.locator(".active-model-line").text_content()
        after = context.request.get(args.url + "/api/v1/sh000001/latest.json")
        assert public.body() == after.body()
        assert not errors, errors
        checks.update(custom_days=17, comparison_export_rows=17, production_release=public.headers["x-model-release"],
                      public_rows=61, json_sha256=hashlib.sha256(public.body()).hexdigest(),
                      shared_dates=60, models=4, prospective_empty=True, page_errors=errors)
        browser.close()
    (args.output / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=True))


if __name__ == "__main__":
    main()
