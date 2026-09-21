"""Prediction service package for immutable, verifiable daily forecasts."""

from .config import Settings
from .web import create_app

__all__ = ["Settings", "create_app"]
