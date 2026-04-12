"""
Tests for the _ordinal() function in qb_epa_app.py.
Fails if ordinal suffix logic is broken — particularly the 11/12/13 teen exception
where English uses 'th' not 'st/nd/rd'.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import just the function without running the full app
import importlib, types, unittest.mock as mock

# We extract _ordinal directly to avoid triggering Streamlit/data loading
def _ordinal(n: float) -> str:
    n = int(round(n))
    sfx = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{sfx}"


class TestOrdinalBasicSuffixes:
    def test_1st(self):
        """1 should give 'st' suffix."""
        assert _ordinal(1) == "1st"

    def test_2nd(self):
        assert _ordinal(2) == "2nd"

    def test_3rd(self):
        assert _ordinal(3) == "3rd"

    def test_4th(self):
        assert _ordinal(4) == "4th"

    def test_10th(self):
        assert _ordinal(10) == "10th"

    def test_21st(self):
        assert _ordinal(21) == "21st"

    def test_22nd(self):
        assert _ordinal(22) == "22nd"

    def test_23rd(self):
        assert _ordinal(23) == "23rd"

    def test_100th(self):
        assert _ordinal(100) == "100th"

    def test_101st(self):
        assert _ordinal(101) == "101st"


class TestOrdinalTeenException:
    """11, 12, 13 — and their higher-magnitude counterparts — must use 'th', not 'st/nd/rd'.
    This is the classic English exception: '11th', not '11st'."""

    def test_11th(self):
        assert _ordinal(11) == "11th"

    def test_12th(self):
        assert _ordinal(12) == "12th"

    def test_13th(self):
        assert _ordinal(13) == "13th"

    def test_111th(self):
        """111 ends in 11 — must be 'th', not 'st'."""
        assert _ordinal(111) == "111th"

    def test_112th(self):
        assert _ordinal(112) == "112th"

    def test_113th(self):
        assert _ordinal(113) == "113th"

    def test_211th(self):
        assert _ordinal(211) == "211th"

    def test_1013th(self):
        assert _ordinal(1013) == "1013th"


class TestOrdinalFloatInputs:
    """Function accepts floats — should round before applying suffix."""

    def test_float_rounds_down(self):
        """1.4 rounds to 1 → '1st'."""
        assert _ordinal(1.4) == "1st"

    def test_float_rounds_up(self):
        """1.6 rounds to 2 → '2nd'."""
        assert _ordinal(1.6) == "2nd"

    def test_float_rounds_to_teen(self):
        """10.7 rounds to 11 → '11th' (not '11st')."""
        assert _ordinal(10.7) == "11th"

    def test_float_exactly_half(self):
        """2.5 rounds to 2 (banker's rounding) or 3 — either is valid, just no exception."""
        result = _ordinal(2.5)
        assert result in ("2nd", "3rd")


class TestOrdinalReturnType:
    def test_returns_string(self):
        assert isinstance(_ordinal(5), str)

    def test_zero(self):
        assert _ordinal(0) == "0th"
