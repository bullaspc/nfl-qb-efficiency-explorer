"""
Shared fixtures for qb_epa_app tests.
Provides a minimal fake DataFrame so tests never hit the NFL data API.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_agg():
    """Minimal aggregated QB DataFrame mimicking what agg looks like after filtering."""
    return pd.DataFrame({
        "QB": ["Patrick Mahomes", "Josh Allen", "Lamar Jackson"],
        "Team": ["KC", "BUF", "BAL"],
        "season": [2024, 2024, 2024],
        "epa_per_play": [0.22, 0.18, 0.15],
        "cpoe": [5.1, 3.8, 2.9],
        "dropbacks": [580, 540, 430],
        "success_rate": [0.52, 0.49, 0.46],
        "epa_clean": [0.25, 0.20, 0.18],
        "epa_pressure": [0.05, 0.02, -0.02],
        "pressure_drop": [0.20, 0.18, 0.20],
        "pressure_rate": [0.24, 0.26, 0.22],
    })


@pytest.fixture
def sample_series():
    """Simple numeric Series for testing _pct_rank behavior."""
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
