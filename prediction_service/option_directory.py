"""Resolve delayed Tushare contract entries against the exchange's own directory."""

from __future__ import annotations

import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from tools.option_features import CONTRACT_FIELDS


SSE_URL = "https://query.sse.com.cn/commonQuery.do"
ACTIVE_QUERY = "SSE_ZQPZ_YSP_GGQQZSXT_XXPL_DRHY_SEARCH_L"
LISTING_QUERY = "SSE_ZQPZ_YSP_OPTZSXT_ADJUST_INFO_HYXG_SEARCH_L"


def _request(parameters):
    request = Request(SSE_URL + "?" + urlencode(parameters), headers={
        "Referer": "https://www.sse.com.cn/assortment/options/disclo/preinfo/",
        "User-Agent": "Mozilla/5.0",
    })
    try:
        with urlopen(request, timeout=15) as response:
            content = response.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("Oversized response")
        payload = json.loads(content)
        if not isinstance(payload, dict) or any(payload.get(key) for key in ("actionErrors", "fieldErrors")):
            raise ValueError("Exchange error")
        if not isinstance(payload.get("result"), list) or not all(isinstance(row, dict) for row in payload["result"]):
            raise ValueError("Invalid directory response")
        return payload
    except Exception:
        raise RuntimeError("SSE option directory request failed; publication retained for retry.") from None


def _active_rows():
    rows, seen = [], set()
    expected = None
    for page in range(1, 101):
        payload = _request({"sqlId": ACTIVE_QUERY, "isPagination": "true", "securityId": "",
                            "expireDate": "", "pageHelp.pageSize": 1000, "pageHelp.pageNo": page,
                            "pageHelp.beginPage": page, "pageHelp.endPage": page, "pageHelp.cacheSize": 1})
        total = payload.get("pageHelp", {}).get("total")
        if not str(total).isdigit() or not 0 < int(total) <= 100000:
            raise ValueError("Invalid SSE option directory pagination.")
        if expected is not None and expected != int(total):
            raise ValueError("SSE option directory changed during pagination; retry required.")
        expected = int(total)
        part = payload["result"]
        codes = [str(row.get("SECURITY_ID", "")) for row in part]
        if not part or len(set(codes)) != len(codes) or seen.intersection(codes):
            raise ValueError("Missing or duplicate SSE option directory page.")
        seen.update(codes)
        rows.extend(part)
        if len(rows) == expected:
            return rows
        if len(rows) > expected:
            raise ValueError("Inconsistent SSE option directory row count.")
    raise ValueError("SSE option directory exceeded pagination bound.")


def _contract(row):
    code = str(row.get("SECURITY_ID", ""))
    identifier = str(row.get("CONTRACT_ID", ""))
    match = re.fullmatch(r"(\d{6})([CP])\d{4}[A-Z]\d{5}", identifier)
    if not re.fullmatch(r"\d{8}", code) or match is None:
        raise ValueError("Invalid SSE contract identifiers.")
    underlying, right = match.groups()
    if row.get("UNDERLYING_SECURITY_ID", underlying) != underlying:
        raise ValueError("Conflicting SSE contract underlying.")
    if {"认购": "C", "认沽": "P", "C": "C", "P": "P"}.get(row.get("CALL_OR_PUT")) != right:
        raise ValueError("Conflicting SSE contract exercise right.")
    dates = []
    for key in ("START_DATE", "END_DATE"):
        value = str(row.get(key, ""))
        if not re.fullmatch(r"\d{8}", value):
            raise ValueError("Missing SSE contract trading dates.")
        dates.append(int(pd.to_datetime(value, format="%Y%m%d", errors="raise").strftime("%Y%m%d")))
    if dates[0] > dates[1]:
        raise ValueError("Invalid SSE contract trading interval.")
    return {"ts_code": code + ".SH", "opt_code": "OP" + underlying + ".SH", "call_put": right,
            "list_date": dates[0], "delist_date": dates[1]}


def complete_option_directory(catalog, observed, remember, *, cached_catalog=None, source_snapshot_id=None):
    """Add only unknown identities; preserve every existing primary-source row."""
    unknown = set(observed.ts_code) - set(catalog.ts_code)
    if not unknown:
        return catalog
    missing = set(unknown)
    additions = []

    if cached_catalog is not None:
        cached = cached_catalog.loc[cached_catalog.ts_code.isin(missing), list(CONTRACT_FIELDS)].copy()
        if cached.ts_code.duplicated().any() or cached.isna().any().any():
            raise ValueError("Invalid archived option directory.")
        for row in cached.to_dict("records"):
            if not re.fullmatch(r"\d{8}\.SH", str(row["ts_code"])) or not re.fullmatch(r"OP\d{6}\.SH", str(row["opt_code"])) or row["call_put"] not in ("C", "P"):
                raise ValueError("Invalid archived option contract identity.")
            observed_dates = pd.to_numeric(observed.loc[observed.ts_code.eq(row["ts_code"]), "trade_date"], errors="raise")
            if not observed_dates.between(int(row["list_date"]), int(row["delist_date"])).all():
                raise ValueError("Archived option dates do not cover observed activity.")
            additions.append(row)
            missing.remove(row["ts_code"])
        if not cached.empty:
            remember("option_cached_directory", cached, provider="frozen_archive", source_snapshot_id=source_snapshot_id)
        if not missing:
            return pd.concat([catalog, pd.DataFrame(additions, columns=CONTRACT_FIELDS)], ignore_index=True)

    def consume(rows, name, query):
        remember(name, pd.DataFrame(rows), provider="SSE", url=SSE_URL, query=query)
        codes = [str(row.get("SECURITY_ID", "")) + ".SH" for row in rows]
        if len(set(codes)) != len(codes):
            raise ValueError("Duplicate SSE contract identities.")
        for row, code in zip(rows, codes, strict=True):
            if code in missing:
                contract = _contract(row)
                dates = pd.to_numeric(observed.loc[observed.ts_code.eq(code), "trade_date"], errors="raise")
                if not dates.between(contract["list_date"], contract["delist_date"]).all():
                    raise ValueError(f"SSE contract dates do not cover observed activity: {code}.")
                additions.append(contract)
                missing.remove(code)

    consume(_active_rows(), "option_sse_active", ACTIVE_QUERY)
    # Contracts can expire during a longer outage. The dated listing records
    # provide the same identity and first/last trading dates after delisting.
    dates = sorted(observed.loc[observed.ts_code.isin(missing), "trade_date"].astype(str).unique())
    for date in dates:
        if not missing:
            break
        payload = _request({"sqlId": LISTING_QUERY, "isPagination": "false", "adjustDate": date, "securityCode": ""})
        rows = payload["result"]
        if any(str(row.get("ADJUST_DATE")) != date for row in rows):
            raise ValueError("Unexpected date in SSE option listing response.")
        consume(rows, f"option_sse_listing_{date}", LISTING_QUERY)
    if missing:
        raise ValueError(f"Option directory not ready; unclassified contracts: {', '.join(sorted(missing)[:10])}.")
    return pd.concat([catalog, pd.DataFrame(additions, columns=CONTRACT_FIELDS)], ignore_index=True)
