# NFL QB Efficiency Explorer

An interactive Streamlit dashboard for exploring NFL quarterback efficiency metrics — EPA per dropback, Completion % Over Expected (CPOE), success rate, and team win-loss records — sourced from [nflfastR](https://www.nflfastr.com/) via `nfl_data_py`.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red)
![Data](https://img.shields.io/badge/data-nflfastR-green)

---

## Features

### League snapshot
Five summary metric cards shown above the tabs: qualifying QB count, best EPA/play, best CPOE, most TDs, and best success rate for the current filter set.

### Global filters (sidebar)
- **Season(s)** — single or multi-season selection (2016–2025)
- **Game type** — Regular Season, Postseason, or Both
- **Week range** — slider for regular-season weeks (disabled for Postseason)
- **Min. pass attempts** — qualifying threshold to filter out small samples (fixed at 5 for Postseason)
- **W-L record display** — choose whether hover tooltips show the Regular Season or Postseason team record

### Five tabs

| Tab | What it shows |
|---|---|
| **▦ EPA Rank** | Horizontal bar chart ranking QBs by EPA per dropback, diverging color scale centered on the league average |
| **◎ EPA vs CPOE** | Scatter plot: efficiency (EPA/play) vs. accuracy (CPOE), bubble size and color encode additional metrics, QB headshots overlaid |
| **⟠ Trends** | Multi-QB season-over-season line chart + single-QB weekly bar/line breakdown |
| **✓ Success Rate** | Horizontal bar chart ranking QBs by % of dropbacks with positive EPA |
| **⊞ Data** | Full aggregated stats table with conditional formatting and CSV download |

### Rich hover tooltips
Every chart shows on hover: QB name, team, season, W-L record, EPA/play, success rate, attempts, TDs, INTs.

---

## Metrics explained

| Metric | Definition |
|---|---|
| **EPA** | Expected Points Added — value of a play relative to a model-estimated baseline |
| **EPA / dropback** | Mean EPA across all pass attempts and QB scrambles |
| **CPOE** | Completion % Over Expected — actual completion rate minus model-predicted rate |
| **Success rate** | % of dropbacks where EPA > 0 |
| **EPA (clean pocket)** | Mean EPA on non-pressured dropbacks |
| **EPA (under pressure)** | Mean EPA on pressured dropbacks |
| **Pressure drop** | EPA clean pocket minus EPA under pressure — measures how much a QB's performance declines when pressured |
| **Pressure rate** | % of dropbacks where the QB faced pressure |
| **Air yards** | Mean air distance of pass attempts (depth of target) |
| **Time to throw** | Mean seconds from snap to release |

Data comes from nflfastR play-by-play, which covers the 2016 season onward.

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install streamlit nfl_data_py pandas plotly
```

> **Note:** `nfl_data_py` downloads data from GitHub on first run; an internet connection is required. Data is cached by Streamlit across reruns.

---

## Running the app

```bash
streamlit run qb_epa_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```
qb_epa_app.py   # single-file Streamlit app — all logic, layout, and charts
README.md
```

### Data flow

```
load_pbp()           nflfastR play-by-play
load_rosters()       Weekly rosters → QB headshot URLs
load_teams()         Team logos and colors
load_qb_records()    Schedule results → per-QB W-L records (games started)
       │
       ▼
Filter dropbacks (pass_attempt == 1 | qb_scramble == 1)
Apply game type / week range / season filters
Compute pressure splits (epa_clean, epa_pressure, pressure_drop)
       │
       ▼
Aggregate per (season, QB, team)
Merge headshots, logos, W-L records
       │
       ▼
agg DataFrame → all five tabs
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | App framework |
| `nfl_data_py` | NFL data (play-by-play, rosters, schedules, team info) |
| `pandas` | Data wrangling |
| `plotly` | Interactive charts |

---

## Design notes

- **Color palette** — RdBu diverging scale throughout: blue = above average / positive, red = below average / negative. Colorblind-safe.
- **Tufte principles** — no chart junk, transparent backgrounds, minimal grid lines, data-ink maximized.
- **Caching** — all data loaders use `@st.cache_data` so switching filters doesn't re-download data.

---

## Data source

Play-by-play data via [nflfastR](https://www.nflfastr.com/), accessed through the [nfl_data_py](https://github.com/cooperdff/nfl_data_py) Python package. Coverage: 2016–present regular season and postseason.
