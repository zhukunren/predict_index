"""Exercise a running local panel without printing administrator credentials."""

from __future__ import annotations

import argparse
import configparser
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--config", type=Path, default=Path("config.ini"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run/service-panel"))
    args = parser.parse_args()
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.config, encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)
    checks = {}
    errors = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url + "/admin/login")
        page.get_by_label("账号", exact=True).fill(config.get("管理员", "账号", fallback="admin"))
        page.get_by_label("密码", exact=True).fill(config.get("管理员", "密码"))
        page.get_by_role("button", name="登录", exact=True).click()
        page.wait_for_url(re.compile(r"/admin/$"))
        page.goto(args.url + "/admin/?days=60")
        page.locator("#returns-chart").wait_for()
        page.wait_for_function("window.Chart && Chart.getChart(document.getElementById('returns-chart'))")
        initial_json = context.request.get(args.url + "/api/v1/sh000001/latest.json")
        assert initial_json.status == 200
        assert initial_json.headers["content-type"].startswith("application/json")
        assert "content-disposition" not in initial_json.headers
        public_payload = initial_json.json()
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
        assert page.locator("[data-direction-diagnostics], [data-direction-alert]").count() == 0
        etag = initial_json.headers["etag"]
        assert context.request.get(args.url + "/api/v1/sh000001/latest.json", headers={"If-None-Match": etag}).status == 304
        for width, height, label in [(1440, 1000, "desktop"), (390, 844, "mobile"), (320, 800, "small-mobile")]:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(300)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), label + " horizontal overflow"
            canvas = page.locator("#returns-chart").evaluate("""canvas => {
                const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
                let colored = 0;
                for (let i = 0; i < pixels.length; i += 16) if (pixels[i+3] > 0 && pixels[i] < 200) colored++;
                return {width: canvas.width, height: canvas.height, colored};
            }""")
            assert canvas["colored"] > 100 and canvas["width"] > 150, label + " blank chart"
            assert page.locator("svg.lucide").count() > 10
            page.screenshot(path=args.output / f"overview-{label}.png", full_page=True)
            checks[label] = canvas
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_label("自定义统计交易日数", exact=True).fill("17")
        page.get_by_role("button", name="应用统计天数").click()
        page.wait_for_url(re.compile(r"days=17"))
        assert len(json.loads(page.locator("#chart-data").text_content())) == 17
        page.locator(".sidebar nav").get_by_role("link", name="历史明细").click()
        assert page.get_by_label("自定义统计交易日数", exact=True).input_value() == "17"
        page.get_by_label("预测方向筛选").select_option("上涨")
        assert all(value == "上涨" for value in page.locator("tbody tr:visible").evaluate_all("rows => rows.map(row => row.dataset.direction)"))
        page.get_by_label("预测方向筛选").select_option("")
        page.locator(".segmented a").filter(has_text="60 日").click()
        page.get_by_role("button", name="下一页", exact=True).click()
        assert page.locator("[data-page]").text_content().startswith("2 /")
        page.screenshot(path=args.output / "history-desktop.png", full_page=True)
        for view in ("jobs", "research"):
            page.goto(args.url + "/admin/?view=" + view)
            assert page.locator("h1").is_visible()
            page.screenshot(path=args.output / (view + "-desktop.png"), full_page=True)
            if view == "research":
                page.set_viewport_size({"width": 320, "height": 800})
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                page.screenshot(path=args.output / "research-small-mobile.png", full_page=True)
                page.set_viewport_size({"width": 1440, "height": 1000})
        page.goto(args.url + "/admin/archives")
        assert page.locator("tbody tr").count() >= 1
        after = context.request.get(args.url + "/api/v1/sh000001/latest.json")
        assert after.headers["etag"] == etag and after.body() == initial_json.body()
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator(".mobile-account").get_by_role("button", name="退出登录").click()
        page.wait_for_url(re.compile(r"/admin/login$"))
        assert not errors, errors
        checks["custom_days"] = 17
        checks["direction_diagnostics_removed"] = True
        checks["json_unchanged"] = True
        checks["page_errors"] = errors
        browser.close()
    (args.output / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
