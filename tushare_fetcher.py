"""English compatibility entry point for :mod:`数据拉取脚本_tushare`."""

from __future__ import annotations

from importlib import import_module


_implementation = import_module("数据拉取脚本_tushare")
__all__ = [name for name in dir(_implementation) if not name.startswith("__")]
globals().update({name: getattr(_implementation, name) for name in __all__})


if __name__ == "__main__":
    _implementation._main()
