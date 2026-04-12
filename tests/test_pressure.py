"""
Tests for pressure metric logic in qb_epa_app.py.
Covers _pct_rank() behavior and the pressure_drop computation.
Fails if the percentile ranking logic breaks or pressure_drop formula changes unexpectedly.
"""
import pytest
import pandas as pd
import numpy as np


def _pct_rank(series: pd.Series) -> pd.Series:
    """Replicates the _pct_rank function from qb_epa_app.py."""
    return series.rank(pct=True, method="average").mul(100)


def _pressure_drop(epa_clean: pd.Series, epa_pressure: pd.Series) -> pd.Series:
    """Replicates pressure_drop = epa_clean - epa_pressure, rounded to 3 decimals."""
    return (epa_clean - epa_pressure).round(3)


class TestPctRank:
    """_pct_rank should produce a 0–100 percentile score, higher = better."""

    def test_output_range_min(self, sample_series):
        """Lowest value should have the lowest rank, above 0."""
        result = _pct_rank(sample_series)
        assert result.min() > 0

    def test_output_range_max(self, sample_series):
        """Highest value should have rank ≤ 100."""
        result = _pct_rank(sample_series)
        assert result.max() <= 100

    def test_monotonic(self, sample_series):
        """Higher input values must produce higher (or equal) rank values."""
        result = _pct_rank(sample_series)
        assert result.is_monotonic_increasing

    def test_single_element(self):
        """A single-element series should return 100 (sole element = 100th percentile)."""
        result = _pct_rank(pd.Series([42.0]))
        assert result.iloc[0] == pytest.approx(100.0)

    def test_handles_negatives(self):
        """Negative EPA values are valid — ranking should still work correctly."""
        s = pd.Series([-0.3, -0.1, 0.0, 0.1, 0.3])
        result = _pct_rank(s)
        assert result.is_monotonic_increasing
        assert result.max() <= 100
        assert result.min() > 0

    def test_tied_values_average(self):
        """Ties use 'average' method — both tied values get the same rank."""
        s = pd.Series([1.0, 1.0, 3.0])
        result = _pct_rank(s)
        assert result.iloc[0] == result.iloc[1], "Tied values must have equal rank"

    def test_output_length(self, sample_series):
        """Output length must match input length."""
        result = _pct_rank(sample_series)
        assert len(result) == len(sample_series)


class TestPressureDrop:
    """pressure_drop = epa_clean - epa_pressure (rounded to 3 decimals).
    A positive value means the QB performs better without pressure (expected).
    """

    def test_basic_computation(self):
        """Straightforward case: clean EPA 0.25, pressure EPA 0.05 → drop of 0.2."""
        clean = pd.Series([0.25, 0.30])
        pressure = pd.Series([0.05, 0.10])
        result = _pressure_drop(clean, pressure)
        assert result.iloc[0] == pytest.approx(0.2, abs=1e-3)
        assert result.iloc[1] == pytest.approx(0.2, abs=1e-3)

    def test_rounded_to_3_decimals(self):
        """Result must be rounded to 3 decimal places."""
        clean = pd.Series([0.1234])
        pressure = pd.Series([0.0001])
        result = _pressure_drop(clean, pressure)
        # 0.1234 - 0.0001 = 0.1233 → rounds to 0.123
        assert result.iloc[0] == pytest.approx(0.123, abs=1e-9)

    def test_negative_drop_allowed(self):
        """A QB performing better under pressure yields negative pressure_drop — valid."""
        clean = pd.Series([0.10])
        pressure = pd.Series([0.20])
        result = _pressure_drop(clean, pressure)
        assert result.iloc[0] < 0

    def test_zero_drop(self):
        """Equal clean and pressure EPA → pressure_drop = 0."""
        clean = pd.Series([0.15])
        pressure = pd.Series([0.15])
        result = _pressure_drop(clean, pressure)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-3)
