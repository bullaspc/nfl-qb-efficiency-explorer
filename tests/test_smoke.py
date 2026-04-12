"""
Smoke tests for qb_epa_app.py using streamlit.testing.v1.AppTest.
Fails if the app raises an unhandled exception on startup or key data structures change.

Note: AppTest runs the app in a sandboxed mode. Because qb_epa_app.py makes
live network calls at module level (load_pbp etc.), we patch those before running.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def make_fake_pbp():
    """
    Minimal play-by-play DataFrame with all columns the app expects (see _PBP_COLS).
    passer_player_id must be a string so the merge with rosters works.
    """
    n = 50
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "season": [2024] * n,
        "week": rng.integers(1, 18, n).tolist(),
        "passer_player_name": ["P.Mahomes"] * 25 + ["J.Allen"] * 25,
        "passer_player_id": ["00-0033873"] * 25 + ["00-0036971"] * 25,
        "pass_attempt": [1] * n,
        "rush_attempt": [0] * n,
        "epa": rng.normal(0.1, 0.3, n),
        "cpoe": rng.normal(2.0, 5.0, n),
        "air_yards": rng.normal(8, 5, n),
        "yards_after_catch": rng.normal(4, 3, n),
        "complete_pass": rng.integers(0, 2, n),
        "interception": [0] * n,
        "touchdown": rng.integers(0, 2, n),
        "sack": [0] * n,
        "qb_scramble": [0] * n,
        "posteam": ["KC"] * 25 + ["BUF"] * 25,
        "defteam": ["BUF"] * 25 + ["KC"] * 25,
        "season_type": ["REG"] * n,
        "was_pressure": rng.integers(0, 2, n).astype(bool),
        "time_to_throw": rng.uniform(1.5, 3.5, n),
        "score_differential": rng.integers(-14, 14, n).astype(float),
        "qtr": rng.integers(1, 5, n),
        "wp": rng.uniform(0.1, 0.9, n),
        "down": rng.integers(1, 5, n).astype(float),
        "ydstogo": rng.integers(1, 20, n).astype(float),
        "game_id": ["2024_01_KC_BUF"] * 25 + ["2024_01_BUF_KC"] * 25,
    })


def make_fake_rosters():
    return pd.DataFrame({
        "season": [2024, 2024],
        "player_id": ["00-0033873", "00-0036971"],
        "headshot_url": ["", ""],
    })


def make_fake_teams():
    return pd.DataFrame({
        "team_abbr": ["KC", "BUF"],
        "team_color": ["#E31837", "#00338D"],
        "team_logo_espn": ["", ""],
    })


def make_fake_schedules():
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 1, 2, 2],
        "game_type": ["REG", "REG", "REG", "REG"],
        "home_team": ["KC", "BUF", "KC", "BUF"],
        "away_team": ["BUF", "KC", "BUF", "KC"],
        "home_score": [27.0, 24.0, 21.0, 17.0],
        "away_score": [24.0, 27.0, 17.0, 21.0],
    })


class TestAppSmoke:
    """Basic smoke tests — does the app start at all?"""

    def test_app_starts_without_exception(self):
        """
        The app should render without raising an unhandled exception.
        Fails if any top-level code crashes, imports are broken, or
        Streamlit layout calls throw an error.
        """
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available (Streamlit < 1.18)")

        with (
            patch("pandas.read_parquet", return_value=make_fake_pbp()),
            patch("nfl_data_py.import_pbp_data", return_value=make_fake_pbp()),
            patch("nfl_data_py.import_weekly_rosters", return_value=make_fake_rosters()),
            patch("nfl_data_py.import_team_desc", return_value=make_fake_teams()),
            patch("nfl_data_py.import_schedules", return_value=make_fake_schedules()),
        ):
            at = AppTest.from_file("qb_epa_app.py", default_timeout=30)
            at.run()

        # The most important assertion: no unhandled crash.
        # AppTest returns ElementList() (not None) when there are no exceptions.
        assert len(at.exception) == 0, f"App raised exception: {at.exception}"

    def test_app_has_no_error_widgets(self):
        """No st.error() calls should appear on a clean load."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        with (
            patch("pandas.read_parquet", return_value=make_fake_pbp()),
            patch("nfl_data_py.import_pbp_data", return_value=make_fake_pbp()),
            patch("nfl_data_py.import_weekly_rosters", return_value=make_fake_rosters()),
            patch("nfl_data_py.import_team_desc", return_value=make_fake_teams()),
            patch("nfl_data_py.import_schedules", return_value=make_fake_schedules()),
        ):
            at = AppTest.from_file("qb_epa_app.py", default_timeout=30)
            at.run()

        assert len(at.error) == 0, f"App showed error widgets: {[e.value for e in at.error]}"
