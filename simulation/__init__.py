"""Reproducible simulation infrastructure for the fuzzy time-series project."""

from .config import Scenario
from .dgp import DGPResult, generate_series

__all__ = ["DGPResult", "Scenario", "generate_series"]
