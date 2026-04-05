# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
streamlit run qb_epa_app.py
```

## Dependencies

Key packages: `streamlit`, `nfl_data_py`, `pandas`, `plotly`

Install with:
```bash
pip install streamlit nfl_data_py pandas plotly
```

## Architecture

This is a single-file Streamlit app ([qb_epa_app.py](qb_epa_app.py)) that visualizes NFL quarterback efficiency metrics sourced from `nfl_data_py` (nflfastR data).

**Data flow:**
1. `load_pbp()` — fetches play-by-play data for selected seasons (cached)
2. `load_rosters()` — fetches QB headshots (cached)
3. `load_teams()` — fetches team logos and colors (cached)
4. `load_qb_records()` — computes per-QB W-L records from schedules (cached)
5. Filtering: dropback plays only (`pass_attempt==1` or `qb_scramble==1`), within selected week/game-type range, with a minimum attempts threshold
6. Pressure splits computed: `epa_clean`, `epa_pressure`, `pressure_drop` per QB
7. Aggregation per QB/season/team → `agg` DataFrame used across all tabs

**Five tabs:**
- **EPA Rankings** — horizontal bar chart, EPA/dropback, diverging color around league average
- **EPA vs CPOE** — scatter with bubble size + headshot images overlaid, quadrant labels
- **Season Trends** — multi-QB line chart + weekly breakdown bar+line combo for a single QB
- **Success Rate** — horizontal bar chart (success = EPA > 0)
- **Data Table** — styled DataFrame with image columns, CSV download

**Chart styling:** `_LAYOUT` dict and `_clean_fig()` apply a shared Tufte-minimal theme (no chart junk, transparent background, RdBu diverging palette) to all `go.Figure` objects. Plotly Express figures call `_clean_fig()` after construction.

**Sidebar controls** (`seasons`, `game_type`, `week_range`, `min_attempts`, `wl_type`) apply globally to all tabs via the `agg` DataFrame filter.
**Deploy** : before merging into main force to do a smoke test.
