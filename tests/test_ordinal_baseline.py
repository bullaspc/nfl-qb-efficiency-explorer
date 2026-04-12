"""
Baseline (no skill) tests for _ordinal() — minimal coverage, generic approach.
"""

def _ordinal(n: float) -> str:
    n = int(round(n))
    sfx = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{sfx}"

def test_ordinal_1():
    assert _ordinal(1) == "1st"

def test_ordinal_2():
    assert _ordinal(2) == "2nd"

def test_ordinal_11():
    assert _ordinal(11) == "11th"

def test_ordinal_13():
    assert _ordinal(13) == "13th"
