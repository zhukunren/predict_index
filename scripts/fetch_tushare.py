"""Run the frozen Tushare data ingestion module."""

from __future__ import annotations

from importlib import import_module


def main() -> None:
    import_module("数据拉取脚本_tushare")._main()


if __name__ == "__main__":
    main()
