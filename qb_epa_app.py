import streamlit as st
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

# ── Palette constants (colorblind-safe) ───────────────────────────────────────
# Diverging: cool blue (good) / warm red (bad), centered at 0
_DIVERG = "RdBu"          # EPA diverging scale  (red = below avg, blue = above)
_DIVERG_R = "RdBu_r"     # reversed for success rate (higher = more blue)
_POS_CLR = "#4575B4"      # blue – above average / positive
_NEG_CLR = "#D73027"      # red  – below average / negative
_NEUTRAL = "#737373"      # mid-gray for reference lines and annotations

st.set_page_config(
    page_title="NFL QB Efficiency — EPA & Success Rate Explorer",
    page_icon="🏈",
    layout="wide",
)

# ── Minimal global style injection ────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Tighten metric card padding */
      [data-testid="metric-container"] { padding: 0.4rem 0.6rem; }

      /* ── Tab navigation ───────────────────────────────────────────────────── */
      /* Breathing room between tab stops; still reads as one strip */
      .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        flex-wrap: wrap;
      }

      /* Larger click target; no background fill (Tufte: no chartjunk) */
      .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.85rem;
        border-radius: 3px 3px 0 0;   /* soften top corners only */
        color: #555555;
        font-size: 0.875rem;
        letter-spacing: 0.01em;
        background: transparent;
        border-bottom: 3px solid transparent;  /* reserve space; hidden when inactive */
        transition: color 0.15s, border-color 0.15s;
      }

      /* Hover: legible highlight, no heavy fill */
      .stTabs [data-baseweb="tab"]:hover {
        color: #111111;
        background: rgba(0,0,0,0.03);
      }

      /* Active tab: clear positional signal via a stronger underline stroke */
      .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1a56db;               /* Streamlit-consistent blue, not a new hue */
        border-bottom: 3px solid #1a56db;
        background: transparent;
        font-weight: 600;             /* bold only on active; see note below */
      }

      /*
        Note on font-weight: bold on the *active* label only is safe here because
        all tab labels are short (1–3 words) and the strip height is fixed by the
        padding — there is no layout shift from the weight change.
      */

      /* ── Responsive radio buttons — wrap on narrow screens ───────────────── */
      .stRadio > div { flex-wrap: wrap; gap: 6px; }

      /* ── Tablet breakpoint (≤768px) ──────────────────────────────────────── */
      @media (max-width: 768px) {
        [data-testid="metric-container"] { padding: 0.3rem 0.4rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.5rem; font-size: 0.75rem; }
        /* Force column blocks to stack vertically */
        [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
      }

      /* ── Phone breakpoint (≤480px) ───────────────────────────────────────── */
      @media (max-width: 480px) {
        [data-testid="metric-container"] { padding: 0.2rem 0.3rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.3rem 0.4rem; font-size: 0.7rem; letter-spacing: 0; }
        h1 { font-size: 1.4rem !important; }
        h2, h3 { font-size: 1.1rem !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NFL QB Efficiency Explorer")
st.caption(
    "Passing efficiency (EPA/dropback) and accuracy (CPOE) from nflfastR · "
    "Positive EPA = play generated value above expectation"
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data filters")

    st.markdown("**Season & schedule**")
    seasons = st.multiselect(
        "Season(s)",
        options=list(range(2016, 2026)),
        default=[2025],
    )
    game_type = st.radio(
        "Game type",
        ["Regular season", "Postseason", "Both"],
        index=0,
        horizontal=True,
    )
    week_range = st.slider("Weeks", 1, 22, (1, 18), help="Regular season = weeks 1–18", disabled=game_type == "Postseason")

    st.markdown("**Qualifying threshold**")
    _postseason = game_type == "Postseason"
    min_attempts = 5 if _postseason else st.slider(
        "Min. pass attempts",
        50, 500, 150, step=25,
        help="QBs below this threshold are excluded from all views",
    )
    if _postseason:
        st.caption("Min. attempts: 5 (postseason).")

    st.markdown("**W-L record display**")
    wl_type = st.radio(
        "Show record for",
        ["Regular Season", "Postseason"],
        index=1 if _postseason else 0,
        horizontal=True,
        key="wl_type",
    )

    st.markdown("**Play filters**")
    quarters = st.multiselect(
        "Quarters", [1, 2, 3, 4], default=[1, 2, 3, 4],
        format_func=lambda q: f"Q{q}",
    )
    wp_range = st.slider(
        "Win probability range", 0.0, 1.0, (0.0, 1.0), step=0.05,
        help="Exclude plays where win probability falls outside this range",
    )
    excl_garbage = st.checkbox(
        "Exclude garbage time", value=False,
        help="Drops plays with score diff >28 or >17 pts in Q4",
    )

if not seasons:
    st.warning("Select at least one season in the sidebar.")
    st.stop()

# ── Disk-based PBP cache ───────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"
_CURRENT_YEAR = 2025
_PBP_COLS = [
    "season", "week", "passer_player_name", "passer_player_id",
    "pass_attempt", "rush_attempt", "epa", "cpoe", "air_yards", "yards_after_catch",
    "complete_pass", "interception", "touchdown", "sack",
    "qb_scramble", "posteam", "defteam", "season_type",
    "was_pressure", "time_to_throw",
    "score_differential", "qtr", "wp",
    "down", "ydstogo", "game_id",
]


def _get_pbp(season: int) -> pd.DataFrame:
    """Return PBP for one season from disk cache; download & save if missing, stale, or schema changed."""
    path = _DATA_DIR / f"pbp_{season}.parquet"
    if path.exists():
        stale = season >= _CURRENT_YEAR and (
            datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) >= timedelta(hours=24)
        )
        if not stale:
            cached = pd.read_parquet(path)
            if all(c in cached.columns for c in _PBP_COLS):
                return cached
    pbp = nfl.import_pbp_data([season], downcast=True)
    for col in _PBP_COLS:
        if col not in pbp.columns:
            pbp[col] = float("nan")
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    pbp[_PBP_COLS].to_parquet(path, index=False)
    return pbp[_PBP_COLS]


# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading play-by-play data…")
def load_pbp(seasons: list[int]) -> pd.DataFrame:
    return pd.concat([_get_pbp(yr) for yr in seasons], ignore_index=True)


@st.cache_data(show_spinner="Loading roster headshots…")
def load_rosters(seasons: list[int]) -> pd.DataFrame:
    # nfl_data_py bug: import_weekly_rosters with multiple seasons produces a
    # duplicate-index DataFrame and then does an in-place assignment that raises
    # ValueError in pandas 2.x. Loading one season at a time sidesteps the issue.
    frames = []
    for yr in seasons:
        try:
            r = nfl.import_weekly_rosters([yr])
            frames.append(
                r[r["position"] == "QB"]
                .dropna(subset=["headshot_url"])
                [["player_id", "season", "headshot_url"]]
            )
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "headshot_url"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "season"])


@st.cache_data(show_spinner=False)
def load_teams() -> pd.DataFrame:
    t = nfl.import_team_desc()[["team_abbr", "team_logo_espn", "team_color"]]
    return t.dropna(subset=["team_logo_espn"])


@st.cache_data(show_spinner="Computing QB game records…")
def load_qb_records(seasons: list[int]) -> pd.DataFrame:
    """W-L record per QB per season, based on games they started (most pass attempts)."""
    _cols = ["season", "week", "season_type", "posteam", "passer_player_name", "pass_attempt"]
    pbp_r = pd.concat([_get_pbp(yr)[_cols] for yr in seasons], ignore_index=True)
    pbp_r = pbp_r[pbp_r["pass_attempt"] == 1].dropna(subset=["passer_player_name"])

    starters = (
        pbp_r.groupby(["season", "week", "season_type", "posteam", "passer_player_name"])["pass_attempt"]
        .sum()
        .reset_index(name="att")
        .sort_values("att", ascending=False)
        .drop_duplicates(["season", "week", "season_type", "posteam"])
        [["season", "week", "season_type", "posteam", "passer_player_name"]]
        .reset_index(drop=True)
    )

    # Schedule results
    sched = nfl.import_schedules(seasons)[
        ["season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"]
    ].dropna(subset=["home_score", "away_score"])
    sched["season_type"] = sched["game_type"].apply(lambda x: "REG" if x == "REG" else "POST")

    base_cols = ["season", "week", "season_type", "home_score", "away_score"]

    home = starters.merge(
        sched[base_cols + ["home_team"]].rename(columns={"home_team": "posteam"}),
        on=["season", "week", "season_type", "posteam"], how="inner",
    )
    home["w"] = (home["home_score"] > home["away_score"]).astype(int)
    home["l"] = (home["home_score"] < home["away_score"]).astype(int)
    home["t"] = (home["home_score"] == home["away_score"]).astype(int)

    away = starters.merge(
        sched[base_cols + ["away_team"]].rename(columns={"away_team": "posteam"}),
        on=["season", "week", "season_type", "posteam"], how="inner",
    )
    away["w"] = (away["away_score"] > away["home_score"]).astype(int)
    away["l"] = (away["away_score"] < away["home_score"]).astype(int)
    away["t"] = (away["away_score"] == away["home_score"]).astype(int)

    keep = ["season", "passer_player_name", "season_type", "w", "l", "t"]
    wl = (
        pd.concat([home[keep], away[keep]], ignore_index=True)
        .groupby(["season", "passer_player_name", "season_type"])[["w", "l", "t"]]
        .sum()
        .reset_index()
    )
    wl["games_started"] = wl["w"] + wl["l"] + wl["t"]
    wl["record"] = wl.apply(
        lambda r: f"{int(r.w)}-{int(r.l)}-{int(r.t)}" if r.t > 0 else f"{int(r.w)}-{int(r.l)}",
        axis=1,
    )
    wl.rename(columns={"passer_player_name": "QB"}, inplace=True)
    return wl[["season", "QB", "season_type", "record", "games_started"]]


@st.cache_data(show_spinner=False)
def load_epa_reference() -> np.ndarray:
    """EPA/play per QB-season from 2016+ (REG, min 150 attempts) — historical percentile pool."""
    ref_cols = ["season", "passer_player_name", "pass_attempt", "epa", "qb_scramble", "season_type"]
    pbp_ref = pd.concat(
        [_get_pbp(yr)[ref_cols] for yr in range(2016, 2026)],
        ignore_index=True,
    )
    pbp_ref = pbp_ref[
        ((pbp_ref["pass_attempt"] == 1) | (pbp_ref["qb_scramble"] == 1)) &
        (pbp_ref["season_type"] == "REG")
    ].dropna(subset=["passer_player_name", "epa"])
    ref_agg = (
        pbp_ref.groupby(["season", "passer_player_name"])
        .agg(att=("pass_attempt", "sum"), epa_per_play=("epa", "mean"))
        .reset_index()
    )
    return ref_agg[ref_agg["att"] >= 150]["epa_per_play"].values


def _ordinal(n: float) -> str:
    n = int(round(n))
    sfx = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{sfx}"


raw = load_pbp(seasons)

# ── Team pass-play weight (pass plays / total offensive plays) ─────────────────
_raw_plays = raw.dropna(subset=["posteam"]).copy()
_raw_plays["_is_pass"] = (_raw_plays["pass_attempt"].fillna(0) == 1).astype(int)
_raw_plays["_is_rush"] = (_raw_plays["rush_attempt"].fillna(0) == 1).astype(int)
_team_pass_weight = (
    _raw_plays.groupby(["season", "posteam"])
    .agg(team_pass_plays=("_is_pass", "sum"), team_rush_plays=("_is_rush", "sum"))
    .reset_index()
)
_team_pass_weight["pass_play_weight"] = (
    _team_pass_weight["team_pass_plays"]
    / (_team_pass_weight["team_pass_plays"] + _team_pass_weight["team_rush_plays"])
).round(3)
_team_pass_weight = (
    _team_pass_weight[["season", "posteam", "pass_play_weight"]]
    .rename(columns={"posteam": "Team"})
)

# ── Team run support: non-QB rushing EPA per play ─────────────────────────────
_team_run_support = (
    _raw_plays[
        (_raw_plays["_is_rush"] == 1) &
        (_raw_plays["qb_scramble"].fillna(0) != 1)
    ]
    .groupby(["season", "posteam"])["epa"]
    .mean()
    .reset_index(name="team_rush_epa")
    .rename(columns={"posteam": "Team"})
)

# ── Filter to dropback plays ───────────────────────────────────────────────────
pbp = raw[
    (raw["pass_attempt"] == 1) | (raw["qb_scramble"] == 1)
].copy()
if game_type == "Regular season":
    pbp = pbp[pbp["season_type"] == "REG"]
    pbp = pbp[pbp["week"].between(week_range[0], week_range[1])].copy()
elif game_type == "Postseason":
    pbp = pbp[pbp["season_type"] == "POST"].copy()
else:
    pbp = pbp[pbp["week"].between(week_range[0], week_range[1])].copy()
pbp = pbp.dropna(subset=["passer_player_name", "epa"])

# ── Play-level filters (quarter / win probability / garbage time) ──────────────
if quarters and set(quarters) != {1, 2, 3, 4}:
    pbp = pbp[pbp["qtr"].isin(quarters)]
if wp_range != (0.0, 1.0):
    pbp = pbp[pbp["wp"].between(wp_range[0], wp_range[1])]
if excl_garbage:
    sd = pbp["score_differential"].fillna(0)
    garbage = (sd.abs() > 28) | ((sd.abs() > 17) & (pbp["qtr"] == 4))
    pbp = pbp[~garbage]

# Success = EPA > 0
pbp["success"] = (pbp["epa"] > 0).astype(float)

# Pressure flag (was_pressure is boolean; coerce robustly)
pbp["pressured"] = pbp["was_pressure"].astype(object) == True

# ── Aggregate per QB ───────────────────────────────────────────────────────────
agg = (
    pbp.groupby(["season", "passer_player_name", "posteam"])
    .agg(
        attempts=("pass_attempt", "sum"),
        dropbacks=("epa", "count"),          # all dropbacks incl. scrambles
        player_id=("passer_player_id", "first"),
        epa_total=("epa", "sum"),
        epa_per_play=("epa", "mean"),
        cpoe=("cpoe", "mean"),
        completion_pct=("complete_pass", "mean"),
        success_rate=("success", "mean"),
        interceptions=("interception", "sum"),
        touchdowns=("touchdown", "sum"),
        sacks=("sack", "sum"),
        air_yards=("air_yards", "mean"),
        pressure_rate=("pressured", "mean"),
        time_to_throw=("time_to_throw", "mean"),
    )
    .reset_index()
)

agg = agg[agg["attempts"] >= min_attempts].copy()
agg = agg.round(2)
agg.rename(columns={"passer_player_name": "QB", "posteam": "Team"}, inplace=True)

# ── Split EPA: clean pocket vs under pressure ──────────────────────────────────
_grp_keys = ["season", "passer_player_name", "posteam"]
_epa_clean = (
    pbp[~pbp["pressured"]].groupby(_grp_keys)["epa"].mean().rename("epa_clean")
)
_epa_press = (
    pbp[pbp["pressured"]].groupby(_grp_keys)["epa"].mean().rename("epa_pressure")
)
_epa_split = (
    pd.concat([_epa_clean, _epa_press], axis=1)
    .reset_index()
    .rename(columns={"passer_player_name": "QB", "posteam": "Team"})
)
_epa_split["pressure_drop"] = (_epa_split["epa_clean"] - _epa_split["epa_pressure"]).round(3)
_epa_split[["epa_clean", "epa_pressure"]] = _epa_split[["epa_clean", "epa_pressure"]].round(3)
agg = agg.merge(
    _epa_split[["season", "QB", "Team", "epa_clean", "epa_pressure", "pressure_drop"]],
    on=["season", "QB", "Team"], how="left",
)

# ── Snap & usage metrics ──────────────────────────────────────────────────────
# Team totals for share calculations
_team_totals = (
    pbp.groupby(["season", "posteam"])
    .agg(team_dropbacks=("epa", "count"), team_epa=("epa", "sum"))
    .reset_index()
    .rename(columns={"posteam": "Team"})
)

# Games played per QB (unique game_ids)
_games = (
    pbp.groupby(["season", "passer_player_name", "posteam"])["game_id"]
    .nunique()
    .reset_index(name="games_played")
    .rename(columns={"passer_player_name": "QB", "posteam": "Team"})
)

# Weekly EPA/dropback std dev (consistency)
_weekly_epa_std = (
    pbp.groupby(["season", "week", "passer_player_name", "posteam"])["epa"]
    .mean()
    .reset_index(name="weekly_epa")
    .groupby(["season", "passer_player_name", "posteam"])["weekly_epa"]
    .std()
    .reset_index(name="weekly_epa_std")
    .rename(columns={"passer_player_name": "QB", "posteam": "Team"})
)

# Passing-down dropback rate (3rd or 4th & ≥5 yards to go)
_pass_down_rate = (
    pbp.assign(is_passing_down=(pbp["down"].isin([3, 4]) & (pbp["ydstogo"] >= 5)).astype(float))
    .groupby(["season", "passer_player_name", "posteam"])["is_passing_down"]
    .mean()
    .reset_index(name="passing_down_rate")
    .rename(columns={"passer_player_name": "QB", "posteam": "Team"})
)

# Join and derive
agg = agg.merge(_team_totals, on=["season", "Team"], how="left")
agg = agg.merge(_games,       on=["season", "QB", "Team"], how="left")
agg = agg.merge(_weekly_epa_std, on=["season", "QB", "Team"], how="left")
agg = agg.merge(_pass_down_rate, on=["season", "QB", "Team"], how="left")

agg["team_dropback_share"] = (agg["dropbacks"] / agg["team_dropbacks"]).round(3)
agg["team_epa_share"]      = (agg["epa_total"]  / agg["team_epa"]).round(3)
agg["snap_adj_epa"]        = (agg["epa_per_play"] * agg["team_dropback_share"]).round(3)
agg["dropbacks_per_game"]  = (agg["dropbacks"] / agg["games_played"]).round(1)
agg["weekly_epa_std"]      = agg["weekly_epa_std"].round(3)
agg["passing_down_rate"]   = agg["passing_down_rate"].round(3)
agg.drop(columns=["team_dropbacks", "team_epa"], inplace=True)

# ── Enrich with headshots & team logos ────────────────────────────────────────
rosters_df = load_rosters(seasons)
teams_df = load_teams()

agg = agg.merge(
    rosters_df, on=["player_id", "season"], how="left"
)
agg = agg.merge(
    teams_df.rename(columns={"team_abbr": "Team"}),
    on="Team", how="left",
)
agg["headshot_url"] = agg["headshot_url"].fillna("")
agg["team_logo_espn"] = agg["team_logo_espn"].fillna("")

# ── Enrich with QB-specific W-L records (games started) ───────────────────────
_qb_wl = load_qb_records(seasons)
_reg = (
    _qb_wl[_qb_wl["season_type"] == "REG"][["season", "QB", "record", "games_started"]]
    .rename(columns={"record": "reg_record", "games_started": "reg_games_started"})
    .drop_duplicates(["season", "QB"])
    .reset_index(drop=True)
)
_post = (
    _qb_wl[_qb_wl["season_type"] == "POST"][["season", "QB", "record", "games_started"]]
    .rename(columns={"record": "post_record", "games_started": "post_games_started"})
    .drop_duplicates(["season", "QB"])
    .reset_index(drop=True)
)
agg = agg.merge(_reg,  on=["season", "QB"], how="left")
agg = agg.merge(_post, on=["season", "QB"], how="left")
agg["reg_record"]  = agg["reg_record"].fillna("—")
agg["post_record"] = agg["post_record"].fillna("—")
agg["reg_games_started"]  = agg["reg_games_started"].fillna(0).astype(int)
agg["post_games_started"] = agg["post_games_started"].fillna(0).astype(int)
# Durability: games started in the relevant context (reg vs post)
agg["games_started"] = (
    agg["post_games_started"] if game_type == "Postseason" else agg["reg_games_started"]
)

# ── Enrich with team pass-play weight ─────────────────────────────────────────
agg = agg.merge(_team_pass_weight, on=["season", "Team"], how="left")

# ── Team run support merge ─────────────────────────────────────────────────────
agg = agg.merge(_team_run_support, on=["season", "Team"], how="left")
agg["team_rush_epa"] = agg["team_rush_epa"].round(3)

# ── Clutch factor: Q4 dropbacks with score within one possession ──────────────
_clutch_mask = (pbp["qtr"] == 4) & (pbp["score_differential"].abs() <= 8)
_clutch_epa = (
    pbp[_clutch_mask]
    .groupby(["season", "passer_player_name", "posteam"])
    .agg(clutch_epa=("epa", "mean"), clutch_dropbacks=("epa", "count"))
    .reset_index()
    .rename(columns={"passer_player_name": "QB", "posteam": "Team"})
)
_clutch_epa["clutch_epa"] = _clutch_epa["clutch_epa"].round(3)
agg = agg.merge(_clutch_epa, on=["season", "QB", "Team"], how="left")

# ── Context Score: weighted percentile index (0–100) ──────────────────────────
def _pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank 0–100 (higher = better) within the current filtered group."""
    return series.rank(pct=True, method="average").mul(100)

agg["_pct_epa"]      = _pct_rank(agg["epa_per_play"])
agg["_pct_pressure"] = _pct_rank(agg["epa_pressure"].fillna(agg["epa_per_play"]))
_clutch_fill = agg["epa_per_play"].where(
    agg["clutch_dropbacks"].fillna(0) < 10, agg["clutch_epa"]
)
agg["_pct_clutch"]  = _pct_rank(_clutch_fill.fillna(agg["epa_per_play"]))
agg["_pct_burden"]  = _pct_rank(agg["pass_play_weight"].fillna(0.5))

# Weights: 40% efficiency + 25% pressure resilience + 20% clutch + 15% pass burden
agg["context_score"] = (
    0.40 * agg["_pct_epa"] +
    0.25 * agg["_pct_pressure"] +
    0.20 * agg["_pct_clutch"] +
    0.15 * agg["_pct_burden"]
).round(1)
agg.drop(columns=["_pct_epa", "_pct_pressure", "_pct_clutch", "_pct_burden"], inplace=True)

# ── Historical EPA percentile (vs all QB-seasons 2016+, REG, min 150 att) ─────
_ref_epa = load_epa_reference()
agg["epa_pct"] = agg["epa_per_play"].apply(
    lambda v: _ordinal(((_ref_epa < v).sum() / len(_ref_epa)) * 100) + " pct"
    if pd.notna(v) and len(_ref_epa) > 0 else "—"
)

agg = agg.reset_index(drop=True)

if agg.empty:
    st.warning("No QBs match the current filters. Try lowering the minimum attempts.")
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
st.subheader("League snapshot")
# Split into two rows (2 + 3) so narrow screens don't squash 5 into tiny columns
_m_row1 = st.columns(2)
_m_row2 = st.columns(3)
c1, c2 = _m_row1
c3, c4, c5 = _m_row2
avg_epa_lg = agg["epa_per_play"].mean()

c1.metric("Qualifying QBs", len(agg))

if agg["epa_per_play"].notna().any():
    top_epa = agg.loc[agg["epa_per_play"].idxmax()]
    c2.metric("Best EPA/play", f"{top_epa['QB']}", delta=f"{top_epa['epa_per_play']:+.2f}")
else:
    c2.metric("Best EPA/play", "N/A")

if agg["cpoe"].notna().any():
    top_cpoe = agg.loc[agg["cpoe"].idxmax()]
    c3.metric("Best CPOE", f"{top_cpoe['QB']}", delta=f"{top_cpoe['cpoe']:+.2f}%")
else:
    c3.metric("Best CPOE", "N/A")

if agg["touchdowns"].notna().any():
    most_tds = agg.loc[agg["touchdowns"].idxmax()]
    c4.metric("Most TDs", f"{most_tds['QB']}", delta=f"{int(most_tds['touchdowns'])} TDs")
else:
    c4.metric("Most TDs", "N/A")

if agg["success_rate"].notna().any():
    top_sr = agg.loc[agg["success_rate"].idxmax()]
    c5.metric("Best Success Rate", f"{top_sr['QB']}", delta=f"{top_sr['success_rate']:.1%}")
else:
    c5.metric("Best Success Rate", "N/A")

# ── Shared layout defaults (Tufte-clean) ──────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#333333"),
    hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#444466", font_size=12,
                    font_color="white"),
    margin=dict(l=5, r=10, t=50, b=30),
    # Remove outer frame
    xaxis=dict(showline=True, linecolor="#cccccc", mirror=False,
               showgrid=False, zeroline=False),
    yaxis=dict(showline=False, showgrid=True,
               gridcolor="rgba(0,0,0,0.06)", zeroline=False),
)


def _clean_fig(fig: go.Figure, **overrides) -> go.Figure:
    """Apply shared Tufte-clean layout to any Figure."""
    layout = {**_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "▦  EPA Rank",       # ranking bars — primary efficiency metric
        "◎  EPA vs CPOE",   # bivariate scatter — efficiency × accuracy
        "⟠  Trends",         # multi-QB longitudinal + weekly breakdown
        "✓  Success Rate",   # secondary ranking — % positive-EPA dropbacks
        "⊞  Data",           # raw table + CSV export
        "◈  Usage",          # snap share & dropback weight metrics
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 – Horizontal bar: EPA per Dropback
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    _all_seasons = sorted(agg["season"].unique(), reverse=True)
    seasons_tab1 = st.multiselect(
        "Season(s)", _all_seasons, default=[_all_seasons[0]], key="bar_season"
    )
    if not seasons_tab1:
        st.info("Select at least one season.")
        st.stop()

    df_bar = agg[agg["season"].isin(seasons_tab1)].sort_values("epa_per_play", ascending=False).reset_index(drop=True)
    df_bar['epa_per_play'] = df_bar['epa_per_play'].round(3)
    _wl_col = "reg_record" if wl_type == "Regular Season" else "post_record"
    _multi = len(seasons_tab1) > 1
    df_bar["_label"] = (
        df_bar["QB"] + "  ·  " + df_bar["Team"] + "  (" + df_bar[_wl_col] + ")"
        + df_bar["season"].apply(lambda s: f"  {s}" if _multi else "")
    )
    df_show = df_bar.sort_values("epa_per_play").reset_index(drop=True)
    lg_avg = df_bar["epa_per_play"].mean()

    # Color: diverge around league average so "average" = white midpoint
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_show["epa_per_play"],
        y=df_show["_label"],
        orientation="h",
        marker=dict(
            color=df_show["epa_per_play"],
            colorscale=_DIVERG,
            cmid=lg_avg,
            showscale=True,
            colorbar=dict(
                title=dict(text="EPA/play", side="right"),
                thickness=8, len=0.4, tickformat="+.2f",
                outlinewidth=0, x=0.99, xanchor="right",
            ),
        ),
        customdata=list(zip(
            df_show["success_rate"].map(lambda v: f"{v:.1%}"),
            df_show["attempts"].map(lambda v: f"{int(v)}"),
            df_show["touchdowns"].map(lambda v: f"{int(v)}"),
            df_show["interceptions"].map(lambda v: f"{int(v)}"),
            df_show["Team"],
            df_show["epa_per_play"].map(lambda v: f"{v:+.2f}"),
            df_show["season"],
            df_show["reg_record"] if wl_type == "Regular Season" else df_show["post_record"],
            df_show["epa_pct"],
        )),
        hovertemplate=(
            f"<span style='font-size:15px'><b>%{{y}}</b></span><br>"
            f"Season: %{{customdata[6]}}<br>"
            "EPA/play: %{customdata[5]}  ·  %{customdata[8]} (since 2016)<br>"
            "Success Rate: %{customdata[0]}<br>"
            "Attempts: %{customdata[1]}<br>"
            "TDs: %{customdata[2]}  ·  INTs: %{customdata[3]}"
            "<extra></extra>"
        ),
        name="",
    ))

    # Team logo images along the right edge
    for _, row in df_show.iterrows():
        if row["team_logo_espn"]:
            fig_bar.add_layout_image(dict(
                source=row["team_logo_espn"],
                x=0.98, y=row["_label"],
                xref="paper", yref="y",
                sizex=0.04, sizey=0.04,
                xanchor="right", yanchor="middle",
                layer="above",
            ))

    # League-average reference line
    fig_bar.add_vline(
        x=lg_avg, line_dash="dot", line_color=_NEUTRAL, opacity=0.7,
        annotation_text=f"Lg avg {lg_avg:+.2f}",
        annotation_position="top right",
        annotation_font_size=9,
        annotation_font_color=_NEUTRAL,
    )

    _clean_fig(
        fig_bar,
        xaxis=dict(
            title="EPA per dropback",
            showline=True, linecolor="#cccccc", showgrid=False, zeroline=True,
            zerolinecolor="#aaaaaa", zerolinewidth=1,
        ),
        yaxis=dict(title="", showline=False, showgrid=False, zeroline=False, tickfont=dict(size=11)),
        title=dict(
            text=(
                f"<b>EPA per Dropback — {', '.join(str(s) for s in sorted(seasons_tab1))}</b>"
                f"<br><sup>Minimum {min_attempts} attempts · "
                f"Blue = above league average · Red = below</sup>"
            ),
            font_size=16, x=0,
        ),
        height=max(350, len(df_show) * 38),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 – Scatter: EPA/play vs CPOE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_ctrl_a, col_ctrl_b = st.columns([1, 1])
    with col_ctrl_a:
        _all_seasons2 = sorted(agg["season"].unique(), reverse=True)
        seasons_tab2 = st.multiselect(
            "Season(s)", _all_seasons2, default=[_all_seasons2[0]], key="scatter_season"
        )
    with col_ctrl_b:
        size_metric = st.selectbox(
            "Bubble size",
            ["attempts", "games_started", "pass_play_weight",
             "touchdowns", "air_yards", "pressure_rate", "time_to_throw"],
            key="scatter_size",
        )

    if not seasons_tab2:
        st.info("Select at least one season.")
    else:
        df_sc = agg[agg["season"].isin(seasons_tab2)].dropna(subset=["cpoe"]).copy()
        if df_sc.empty:
            st.info("No CPOE data available for the selected season / filters.")
        else:
            # Pre-format all numeric columns — Plotly d3 specifiers are unreliable on customdata
            _fmt = lambda v, spec: format(v, spec) if pd.notna(v) else "—"
            df_sc["_epa_fmt"]        = df_sc["epa_per_play"].map(lambda v: _fmt(v, "+.2f"))
            df_sc["_epa_clean_fmt"]  = df_sc["epa_clean"].map(lambda v: _fmt(v, "+.2f"))
            df_sc["_epa_press_fmt"]  = df_sc["epa_pressure"].map(lambda v: _fmt(v, "+.2f"))
            df_sc["_pdrop_fmt"]      = df_sc["pressure_drop"].map(lambda v: _fmt(v, ".2f"))
            df_sc["_cpoe_fmt"]       = df_sc["cpoe"].map(lambda v: _fmt(v, "+.2f"))
            df_sc["_sr_fmt"]         = df_sc["success_rate"].map(lambda v: _fmt(v, ".1%"))
            df_sc["_comp_fmt"]       = df_sc["completion_pct"].map(lambda v: _fmt(v, ".1%"))
            df_sc["_ay_fmt"]         = df_sc["air_yards"].map(lambda v: _fmt(v, ".1f"))
            df_sc["_prate_fmt"]      = df_sc["pressure_rate"].map(lambda v: _fmt(v, ".1%"))
            df_sc["_ttt_fmt"]        = df_sc["time_to_throw"].map(lambda v: _fmt(v, ".2f"))
            df_sc["_att_fmt"]        = df_sc["attempts"].map(lambda v: f"{int(v)}" if pd.notna(v) else "—")
            df_sc["_td_fmt"]         = df_sc["touchdowns"].map(lambda v: f"{int(v)}" if pd.notna(v) else "—")
            df_sc["_int_fmt"]        = df_sc["interceptions"].map(lambda v: f"{int(v)}" if pd.notna(v) else "—")

            avg_epa_sc = df_sc["epa_per_play"].mean()
            avg_cpoe_sc = df_sc["cpoe"].mean()
            _sc_seasons_str = ", ".join(str(s) for s in sorted(seasons_tab2))

            fig_sc = px.scatter(
                df_sc,
                x="cpoe",
                y="epa_per_play",
                text="QB",
                size=size_metric,
                color="success_rate",
                color_continuous_scale=_DIVERG_R,
                color_continuous_midpoint=df_sc["success_rate"].mean(),
                custom_data=["Team",
                             "_att_fmt", "_td_fmt", "_int_fmt",
                             "_comp_fmt", "_sr_fmt", "_ay_fmt",
                             "season",
                             "reg_record" if wl_type == "Regular Season" else "post_record",
                             "_epa_clean_fmt", "_epa_press_fmt", "_pdrop_fmt",
                             "_prate_fmt", "_ttt_fmt", "_epa_fmt", "_cpoe_fmt"],
                labels={
                    "cpoe": "Completion % Over Expected (CPOE)",
                    "epa_per_play": "EPA per Dropback",
                    "success_rate": "Success Rate",
                },
                height=500,
            )
            fig_sc.update_traces(
                textposition="top center",
                texttemplate="%{text} · %{customdata[0]}<br><sup>%{customdata[7]}  %{customdata[8]}</sup>",
                textfont=dict(size=10, color="#111111"),
                marker=dict(opacity=0.55, line=dict(width=0.5, color="white")),
                hovertemplate=(
                    "<span style='font-size:16px'><b>%{text} · %{customdata[0]}</b></span><br>"
                    f"Season: %{{customdata[7]}}  ·  {wl_type} Record: %{{customdata[8]}}<br>"
                    "EPA/play: %{customdata[14]}  ·  CPOE: %{customdata[15]}%<br>"
                    "Success Rate: %{customdata[5]}<br>"
                    "Attempts: %{customdata[1]}  ·  TDs: %{customdata[2]}  ·  INTs: %{customdata[3]}<br>"
                    "Comp%: %{customdata[4]}  ·  AvgAY: %{customdata[6]}<br>"
                    "<b>── Pressure ──</b><br>"
                    "Clean EPA: %{customdata[9]}  ·  Pressure EPA: %{customdata[10]}<br>"
                    "Pressure Drop: %{customdata[11]}  ·  Pressure Rate: %{customdata[12]}<br>"
                    "Time to Throw: %{customdata[13]}s"
                    "<extra></extra>"
                ),
            )

            # Headshot images at each QB's position on the scatter
            # Reference lines at league averages
            fig_sc.add_hline(
                y=avg_epa_sc, line_dash="dot", line_color=_NEUTRAL, opacity=0.6,
                annotation_text=f"Avg EPA {avg_epa_sc:+.2f}",
                annotation_position="bottom right",
                annotation_font_size=11, annotation_font_color=_NEUTRAL,
            )
            fig_sc.add_vline(
                x=avg_cpoe_sc, line_dash="dot", line_color=_NEUTRAL, opacity=0.6,
                annotation_text=f"Avg CPOE {avg_cpoe_sc:+.2f}%",
                annotation_position="top left",
                annotation_font_size=11, annotation_font_color=_NEUTRAL,
            )

            x_lo, x_hi = df_sc["cpoe"].min(), df_sc["cpoe"].max()
            y_lo, y_hi = df_sc["epa_per_play"].min(), df_sc["epa_per_play"].max()
            qx_right = (avg_cpoe_sc + x_hi) / 2
            qx_left  = (avg_cpoe_sc + x_lo) / 2
            qy_top   = (avg_epa_sc  + y_hi) / 2
            qy_bot   = (avg_epa_sc  + y_lo) / 2

            for label, qx, qy in [
                ("Accurate &amp; Efficient", qx_right, qy_top),
                ("Accurate, Lower Value",    qx_left,  qy_top),
                ("High EPA, Lower Accuracy", qx_right, qy_bot),
                ("Below Average",            qx_left,  qy_bot),
            ]:
                fig_sc.add_annotation(
                    x=qx, y=qy, text=label, showarrow=False,
                    font=dict(size=11, color=_NEUTRAL), opacity=0.5,
                    xanchor="center", yanchor="middle",
                )

            fig_sc.update_layout(
                title=dict(
                    text=f"<b>Accuracy vs Efficiency — {_sc_seasons_str}</b>"
                         f"<br><sup>CPOE: completion % relative to model expectation · "
                         f"Color = success rate · Size = {size_metric}</sup>",
                    font_size=16, x=0,
                ),
                coloraxis_colorbar=dict(
                    title=dict(text="Success Rate", side="right"),
                    tickformat=".0%", thickness=8, len=0.4, outlinewidth=0,
                    x=0.99, xanchor="right",
                ),
            )
            _clean_fig(
                fig_sc,
                xaxis=dict(
                    title="Completion % Over Expected (CPOE)",
                    showline=True, linecolor="#cccccc", showgrid=False,
                    zeroline=True, zerolinecolor="#aaaaaa", zerolinewidth=1,
                ),
                yaxis=dict(
                    title="EPA per Dropback", showline=False,
                    showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                ),
            )
            st.plotly_chart(fig_sc, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 – Season trends + weekly breakdown
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    all_qbs = sorted(agg["QB"].unique())
    default_qbs = agg.groupby("QB")["epa_per_play"].mean().nlargest(5).index.tolist()

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        selected_qbs = st.multiselect(
            "QBs to compare", all_qbs, default=default_qbs, key="trend_qbs"
        )
    with col_t2:
        metric_opts = {
            "EPA / Dropback": "epa_per_play",
            "Success Rate": "success_rate",
            "CPOE (%)": "cpoe",
            "Completion %": "completion_pct",
        }
        metric_label = st.selectbox("Metric", list(metric_opts.keys()), key="trend_metric")
    metric_col = metric_opts[metric_label]

    if selected_qbs:
        df_trend = agg[agg["QB"].isin(selected_qbs)]
        is_pct = metric_col in ("success_rate", "completion_pct")

        # Qualitative palette — 10 distinct, colorblind-tolerant colors (Tableau 10)
        fig_trend = px.line(
            df_trend,
            x="season",
            y=metric_col,
            color="QB",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
            hover_data={"attempts": True, "touchdowns": True, "success_rate": ":.1%"},
            labels={metric_col: metric_label, "season": "Season"},
            height=420,
        )
        fig_trend.update_traces(line=dict(width=2), marker=dict(size=7))
        if is_pct:
            fig_trend.update_layout(yaxis_tickformat=".0%")

        if len(seasons) == 1:
            season_range_str = str(seasons[0])
            trend_title_suffix = f"Season {season_range_str}"
        else:
            season_range_str = f"{min(seasons)}–{max(seasons)}"
            trend_title_suffix = f"Seasons {season_range_str}"
        _clean_fig(
            fig_trend,
            title=dict(
                text=f"<b>{metric_label} — Season Trend</b>"
                     f"<br><sup>{trend_title_suffix} · "
                     f"min {min_attempts} attempts per season</sup>",
                font_size=14, x=0,
            ),
            xaxis=dict(
                title="Season", dtick=1,
                showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
            ),
            yaxis=dict(
                title=metric_label, showline=False,
                showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                title=dict(text=""), font=dict(size=11),
            ),
        )
        st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info("Select at least one QB above.")

    st.markdown("---")
    st.markdown("**Weekly breakdown for a single QB**")
    col_a, col_b = st.columns(2)
    with col_a:
        # Default to the first QB currently selected in the trend chart (if any)
        _weekly_default_idx = (
            all_qbs.index(selected_qbs[0])
            if selected_qbs and selected_qbs[0] in all_qbs
            else 0
        )
        qb_weekly = st.selectbox("QB", all_qbs, index=_weekly_default_idx, key="weekly_qb")
    with col_b:
        season_w = st.selectbox(
            "Season", sorted(agg["season"].unique(), reverse=True), key="weekly_season"
        )

    weekly = (
        pbp[
            (pbp["passer_player_name"] == qb_weekly)
            & (pbp["season"] == season_w)
        ]
        .groupby("week")
        .agg(
            epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"),
            attempts=("pass_attempt", "sum"),
            opponent=("defteam", "first"),
        )
        .reset_index()
    )

    if weekly.empty:
        st.info("No data for this QB / season combination.")
    else:
        _qb_row = agg[(agg["QB"] == qb_weekly) & (agg["season"] == season_w)]
        if not _qb_row.empty:
            _r = _qb_row.iloc[0]
            _rec = _r["reg_record"] if wl_type == "Regular Season" else _r["post_record"]
            st.caption(
                f"**{qb_weekly}** · {_r['Team']} · {season_w} · "
                f"{wl_type} record: **{_rec}**"
            )
        fig_week = go.Figure()
        fig_week.add_trace(go.Bar(
            x=weekly["week"],
            y=weekly["epa_per_play"],
            name="EPA/play",
            marker=dict(
                color=weekly["epa_per_play"],
                colorscale=_DIVERG,
                cmid=0,
                showscale=False,
                opacity=0.85,
            ),
            customdata=(
                weekly.assign(
                    _epa_fmt=weekly["epa_per_play"].map(lambda v: f"{v:+.2f}"),
                    _sr_fmt=weekly["success_rate"].map(lambda v: f"{v:.1%}"),
                )[["_epa_fmt", "_sr_fmt", "opponent", "attempts"]].values
            ),
            hovertemplate=(
                "<b>Week %{x}  vs  %{customdata[2]}</b><br>"
                "EPA/play: %{customdata[0]}<br>"
                "Success Rate: %{customdata[1]}<br>"
                "Attempts: %{customdata[3]}"
                "<extra></extra>"
            ),
            # Show attempt count above/below each bar so small-sample weeks are obvious
            text=weekly["attempts"],
            textposition="outside",
            textfont=dict(size=8, color="#888888"),
        ))
        fig_week.add_trace(go.Scatter(
            x=weekly["week"],
            y=weekly["success_rate"],
            name="Success Rate",
            mode="lines+markers",
            line=dict(color=_POS_CLR, width=2),
            marker=dict(size=6, symbol="circle"),
            yaxis="y2",
            customdata=weekly["success_rate"].map(lambda v: f"{v:.1%}").values,
            hovertemplate="Week %{x}  ·  Success Rate: %{customdata}<extra></extra>",
        ))

        # Dynamic success-rate axis range: 10-pp padding beyond actual data,
        # clamped to [0, 1] so clipping never occurs
        sr_lo = max(0.0, weekly["success_rate"].min() - 0.10)
        sr_hi = min(1.0, weekly["success_rate"].max() + 0.10)

        # Flag any low-sample weeks (< 10 attempts) for the reader
        low_sample_weeks = weekly[weekly["attempts"] < 10]
        if not low_sample_weeks.empty:
            st.caption(
                f"Note: weeks {', '.join(str(w) for w in low_sample_weeks['week'])} "
                f"have fewer than 10 attempts — treat those bars with caution."
            )

        _clean_fig(
            fig_week,
            title=dict(
                text=f"<b>{qb_weekly} — Weekly Performance, {season_w}</b>"
                     f"<br><sup>Bars = EPA/dropback · number above bar = attempts · "
                     f"line = success rate (right axis)</sup>",
                font_size=13, x=0,
            ),
            xaxis=dict(
                title="Week", dtick=1,
                showline=True, linecolor="#cccccc", showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="EPA per Dropback",
                showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=True,
                zerolinecolor="#aaaaaa", zerolinewidth=1,
            ),
            yaxis2=dict(
                title="Success Rate", overlaying="y", side="right",
                tickformat=".0%", range=[sr_lo, sr_hi], showgrid=False,
                zeroline=False, showline=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
            height=360,
        )
        st.plotly_chart(fig_week, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 – Success Rate ranking (moved from Tab 2 to reduce up-front noise)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    _all_seasons4 = sorted(agg["season"].unique(), reverse=True)
    seasons_tab4 = st.multiselect(
        "Season(s)", _all_seasons4, default=[_all_seasons4[0]], key="sr_season"
    )
    if not seasons_tab4:
        st.info("Select at least one season.")
        st.stop()

    df_sr = agg[agg["season"].isin(seasons_tab4)].sort_values("success_rate", ascending=False).reset_index(drop=True)
    _wl_col_sr = "reg_record" if wl_type == "Regular Season" else "post_record"
    _multi4 = len(seasons_tab4) > 1
    df_sr["_label"] = (
        df_sr["QB"] + "  ·  " + df_sr["Team"] + "  (" + df_sr[_wl_col_sr] + ")"
        + df_sr["season"].apply(lambda s: f"  {s}" if _multi4 else "")
    )
    df_sr_show = df_sr.sort_values("success_rate").reset_index(drop=True)
    league_avg_sr = df_sr["success_rate"].mean()

    fig_sr = go.Figure()
    fig_sr.add_trace(go.Bar(
        x=df_sr_show["success_rate"],
        y=df_sr_show["_label"],
        orientation="h",
        marker=dict(
            color=df_sr_show["success_rate"],
            colorscale=_DIVERG_R,
            cmid=league_avg_sr,
            showscale=True,
            colorbar=dict(
                title=dict(text="Success Rate", side="right"),
                thickness=8, len=0.4, tickformat=".0%", outlinewidth=0,
                x=0.99, xanchor="right",
            ),
        ),
        customdata=list(zip(
            df_sr_show["epa_per_play"].map(lambda v: f"{v:+.2f}"),
            df_sr_show["attempts"].map(lambda v: f"{int(v)}"),
            df_sr_show["touchdowns"].map(lambda v: f"{int(v)}"),
            df_sr_show["interceptions"].map(lambda v: f"{int(v)}"),
            df_sr_show["Team"],
            df_sr_show["success_rate"].map(lambda v: f"{v:.1%}"),
            df_sr_show["season"],
            df_sr_show["reg_record"] if wl_type == "Regular Season" else df_sr_show["post_record"],
        )),
        hovertemplate=(
            "<span style='font-size:15px'><b>%{y}</b></span><br>"
            f"Season: %{{customdata[6]}}<br>"
            "Success Rate: %{customdata[5]}<br>"
            "EPA/play: %{customdata[0]}<br>"
            "Attempts: %{customdata[1]}<br>"
            "TDs: %{customdata[2]}  ·  INTs: %{customdata[3]}"
            "<extra></extra>"
        ),
        name="",
    ))

    # Team logo images along the right edge
    for _, row in df_sr_show.iterrows():
        if row["team_logo_espn"]:
            fig_sr.add_layout_image(dict(
                source=row["team_logo_espn"],
                x=0.98, y=row["_label"],
                xref="paper", yref="y",
                sizex=0.04, sizey=0.04,
                xanchor="right", yanchor="middle",
                layer="above",
            ))

    fig_sr.add_vline(
        x=league_avg_sr, line_dash="dot", line_color=_NEUTRAL, opacity=0.7,
        annotation_text=f"Lg avg {league_avg_sr:.1%}",
        annotation_position="top right",
        annotation_font_size=9, annotation_font_color=_NEUTRAL,
    )

    _clean_fig(
        fig_sr,
        title=dict(
            text=(
                f"<b>Success Rate — {', '.join(str(s) for s in sorted(seasons_tab4))}</b>"
                f"<br><sup>% of dropbacks generating positive EPA · "
                f"minimum {min_attempts} attempts</sup>"
            ),
            font_size=14, x=0,
        ),
        xaxis=dict(
            title="Success Rate (% positive-EPA dropbacks)",
            tickformat=".0%",
            showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
        ),
        yaxis=dict(title="", showline=False, showgrid=False, zeroline=False, tickfont=dict(size=11)),
        height=max(350, len(df_sr_show) * 38),
        showlegend=False,
    )
    st.plotly_chart(fig_sr, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 – Raw data table
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    _all_seasons5 = sorted(agg["season"].unique(), reverse=True)
    seasons_tab5 = st.multiselect(
        "Season(s)", _all_seasons5, default=_all_seasons5, key="data_season"
    )
    agg_tab5 = agg[agg["season"].isin(seasons_tab5)] if seasons_tab5 else agg
    st.markdown("**Aggregated QB stats** — sorted by season then EPA/play")

    _all_cols = [
        "headshot_url", "team_logo_espn",
        "season", "QB", "Team", "attempts", "dropbacks",
        "epa_per_play", "epa_total", "success_rate",
        "cpoe", "completion_pct", "touchdowns", "interceptions", "sacks", "air_yards",
        "epa_clean", "epa_pressure", "pressure_drop", "pressure_rate", "time_to_throw",
        "dropbacks_per_game", "team_dropback_share", "team_epa_share",
        "snap_adj_epa", "weekly_epa_std", "passing_down_rate",
    ]
    _core_cols = [
        "headshot_url", "team_logo_espn",
        "season", "QB", "Team", "attempts",
        "epa_per_play", "success_rate", "cpoe", "epa_clean", "pressure_drop",
    ]
    _optional_cols = [c for c in _all_cols if c not in _core_cols]

    with st.expander("Select columns to display", expanded=False):
        extra_cols = st.multiselect(
            "Additional columns",
            options=_optional_cols,
            default=[],
            key="data_extra_cols",
        )
    cols_display = _core_cols + extra_cols

    _all_fmt = {
        "attempts":             "{:.0f}",
        "dropbacks":            "{:.0f}",
        "touchdowns":           "{:.0f}",
        "interceptions":        "{:.0f}",
        "sacks":                "{:.0f}",
        "epa_per_play":         "{:+.2f}",
        "epa_total":            "{:+.1f}",
        "success_rate":         "{:.1%}",
        "completion_pct":       "{:.1%}",
        "cpoe":                 "{:+.2f}",
        "air_yards":            "{:.1f}",
        "epa_clean":            "{:+.2f}",
        "epa_pressure":         "{:+.2f}",
        "pressure_drop":        "{:.2f}",
        "pressure_rate":        "{:.1%}",
        "time_to_throw":        "{:.2f}s",
        "dropbacks_per_game":   "{:.1f}",
        "team_dropback_share":  "{:.1%}",
        "team_epa_share":       "{:.1%}",
        "snap_adj_epa":         "{:+.2f}",
        "weekly_epa_std":       "{:.2f}",
        "passing_down_rate":    "{:.1%}",
    }
    _active_fmt = {k: v for k, v in _all_fmt.items() if k in cols_display}
    _center_cols = [c for c in cols_display if c not in ("headshot_url", "team_logo_espn", "QB", "Team")]
    _df_styled = agg_tab5[cols_display].sort_values(["season", "epa_per_play"], ascending=[False, False])
    _styler = _df_styled.style.format(_active_fmt, na_rep="—")
    if "epa_per_play" in cols_display:
        _styler = _styler.background_gradient(subset=["epa_per_play"], cmap="RdBu", vmin=-0.3, vmax=0.3)
    if "success_rate" in cols_display:
        _styler = _styler.background_gradient(subset=["success_rate"], cmap="RdBu", vmin=0.35, vmax=0.65)
    if "pressure_drop" in cols_display:
        _styler = _styler.background_gradient(subset=["pressure_drop"], cmap="RdBu_r", vmin=0.2, vmax=1.2)
    styled = _styler.set_properties(subset=_center_cols, **{"text-align": "center"})

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "headshot_url":   st.column_config.ImageColumn("Photo", width="small"),
            "team_logo_espn": st.column_config.ImageColumn("Logo",  width="small"),
        },
    )
    cols_export = [c for c in _all_cols if c not in ("headshot_url", "team_logo_espn")]
    csv = agg_tab5[cols_export].to_csv(index=False)
    
    st.download_button("Download CSV", csv, "qb_epa.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 – QB Context & Pressure
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    _all_seasons6 = sorted(agg["season"].unique(), reverse=True)
    seasons_tab6 = st.multiselect(
        "Season(s)", _all_seasons6, default=[_all_seasons6[0]], key="usage_season"
    )
    if not seasons_tab6:
        st.info("Select at least one season.")
        st.stop()

    df_usage = (
        agg[agg["season"].isin(seasons_tab6)]
        .dropna(subset=["epa_per_play"])
        .reset_index(drop=True)
    )

    st.markdown(
        "**Pressure drop** = EPA clean pocket − EPA under pressure (lower = more resilient) · "
        "**Clutch EPA** = EPA/dropback in Q4 with score within 8 pts · "
        "**Context Score** = 40% EPA + 25% pressure resilience + 20% clutch + 15% pass burden (percentile-weighted, 0–100)"
    )

    # ── ROW 1: Pressure Analysis ───────────────────────────────────────────────
    st.markdown("#### Pressure Analysis")
    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        # Horizontal bar: pressure_drop ranking
        df_press = (
            df_usage.dropna(subset=["pressure_drop"])
            .sort_values("pressure_drop", ascending=True)
            .reset_index(drop=True)
        )
        fig_pdrop = go.Figure(go.Bar(
            x=df_press["pressure_drop"],
            y=df_press["QB"].str.split(".").str[-1],
            orientation="h",
            marker=dict(
                color=df_press["pressure_drop"],
                colorscale=_DIVERG,
                cmid=0,
                reversescale=True,
                showscale=False,
            ),
            customdata=list(zip(
                df_press["QB"],
                df_press["epa_clean"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
                df_press["epa_pressure"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
                df_press["pressure_drop"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
                df_press["pressure_rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Clean EPA: %{customdata[1]}<br>"
                "Pressure EPA: %{customdata[2]}<br>"
                "Drop: %{customdata[3]}<br>"
                "Pressure rate: %{customdata[4]}"
                "<extra></extra>"
            ),
            name="",
        ))
        fig_pdrop.add_vline(x=0, line_dash="solid", line_color=_NEUTRAL, opacity=0.4)
        _clean_fig(
            fig_pdrop,
            title=dict(
                text="<b>Pressure Resilience</b>"
                     "<br><sup>Clean EPA − Pressure EPA · lower bar = less drop-off</sup>",
                font_size=13, x=0,
            ),
            xaxis=dict(
                title="Pressure Drop (EPA units)", tickformat="+.3f",
                showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
            ),
            yaxis=dict(title="", showline=False, showgrid=False, zeroline=False, tickfont=dict(size=10)),
            height=max(300, len(df_press) * 24),
            showlegend=False,
            margin=dict(l=5, r=10, t=50, b=30),
        )
        st.plotly_chart(fig_pdrop, width="stretch")

    with col_p2:
        # Scatter: pressure_rate (X) vs epa_pressure (Y)
        df_ps = df_usage.dropna(subset=["pressure_rate", "epa_pressure"]).reset_index(drop=True)
        avg_prate = df_ps["pressure_rate"].mean()
        avg_epa_press = df_ps["epa_pressure"].mean()

        fig_ps = go.Figure()
        fig_ps.add_trace(go.Scatter(
            x=df_ps["pressure_rate"],
            y=df_ps["epa_pressure"],
            mode="markers+text",
            text=df_ps["QB"].str.split(".").str[-1],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=10,
                color=df_ps["pressure_drop"],
                colorscale=_DIVERG,
                cmid=df_ps["pressure_drop"].median(),
                reversescale=True,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Pressure Drop", side="right"),
                    thickness=8, len=0.4, tickformat="+.3f", outlinewidth=0,
                    x=0.99, xanchor="right",
                ),
                line=dict(width=0.5, color="#555555"),
            ),
            customdata=list(zip(
                df_ps["QB"],
                df_ps["pressure_rate"].map(lambda v: f"{v:.1%}"),
                df_ps["epa_pressure"].map(lambda v: f"{v:+.3f}"),
                df_ps["pressure_drop"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pressure rate: %{customdata[1]}<br>"
                "EPA under pressure: %{customdata[2]}<br>"
                "Pressure drop: %{customdata[3]}"
                "<extra></extra>"
            ),
            name="",
        ))
        fig_ps.add_vline(x=avg_prate, line_dash="dot", line_color=_NEUTRAL, opacity=0.5,
                         annotation_text="avg", annotation_position="top right",
                         annotation_font_size=8, annotation_font_color=_NEUTRAL)
        fig_ps.add_hline(y=avg_epa_press, line_dash="dot", line_color=_NEUTRAL, opacity=0.5)

        _clean_fig(
            fig_ps,
            title=dict(
                text="<b>Pressure Rate vs EPA Under Pressure</b>"
                     "<br><sup>Top-left = elite · bottom-right = vulnerable · color = pressure drop</sup>",
                font_size=13, x=0,
            ),
            xaxis=dict(
                title="Pressure Rate", tickformat=".0%",
                showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
            ),
            yaxis=dict(
                title="EPA/dropback Under Pressure",
                showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
            ),
            height=440,
            showlegend=False,
        )
        st.plotly_chart(fig_ps, width="stretch")

    # ── ROW 2: Team Scheme Context + Clutch Factor ─────────────────────────────
    st.markdown("#### Team Context & Clutch")
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        # Scatter: pass_play_weight (X) vs team_rush_epa (Y)
        df_scheme = df_usage.dropna(subset=["pass_play_weight", "team_rush_epa"]).reset_index(drop=True)
        avg_ppw = df_scheme["pass_play_weight"].mean()
        avg_rush = df_scheme["team_rush_epa"].mean()

        fig_scheme = go.Figure()
        fig_scheme.add_trace(go.Scatter(
            x=df_scheme["pass_play_weight"],
            y=df_scheme["team_rush_epa"],
            mode="markers+text",
            text=df_scheme["QB"].str.split(".").str[-1],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=df_scheme["team_dropback_share"].clip(0).mul(40).add(6),
                color=df_scheme["epa_per_play"],
                colorscale=_DIVERG,
                cmid=df_scheme["epa_per_play"].mean(),
                showscale=True,
                colorbar=dict(
                    title=dict(text="QB EPA/play", side="right"),
                    thickness=8, len=0.4, tickformat="+.2f", outlinewidth=0,
                    x=0.99, xanchor="right",
                ),
                line=dict(width=0.5, color="#555555"),
            ),
            customdata=list(zip(
                df_scheme["QB"],
                df_scheme["Team"],
                df_scheme["pass_play_weight"].map(lambda v: f"{v:.1%}"),
                df_scheme["team_rush_epa"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
                df_scheme["epa_per_play"].map(lambda v: f"{v:+.3f}"),
                df_scheme["team_dropback_share"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Pass heaviness: %{customdata[2]}<br>"
                "Team rush EPA: %{customdata[3]}<br>"
                "QB EPA/play: %{customdata[4]}<br>"
                "Dropback share: %{customdata[5]}"
                "<extra></extra>"
            ),
            name="",
        ))
        fig_scheme.add_vline(x=avg_ppw, line_dash="dot", line_color=_NEUTRAL, opacity=0.5,
                             annotation_text="avg pass%", annotation_position="top right",
                             annotation_font_size=8, annotation_font_color=_NEUTRAL)
        fig_scheme.add_hline(y=avg_rush, line_dash="dot", line_color=_NEUTRAL, opacity=0.5,
                             annotation_text="avg rush EPA", annotation_position="right",
                             annotation_font_size=8, annotation_font_color=_NEUTRAL)

        _clean_fig(
            fig_scheme,
            title=dict(
                text="<b>Team Scheme Context</b>"
                     "<br><sup>Pass heaviness × run support · bubble = dropback share · color = QB EPA/play</sup>",
                font_size=13, x=0,
            ),
            xaxis=dict(
                title="Team Pass Play %", tickformat=".0%",
                showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
            ),
            yaxis=dict(
                title="Team Rush EPA/play (non-QB)",
                showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
            ),
            height=440,
            showlegend=False,
        )
        st.plotly_chart(fig_scheme, width="stretch")

    with col_c2:
        # Horizontal bar: clutch_epa ranking
        df_clutch = (
            df_usage.dropna(subset=["clutch_epa"])
            .sort_values("clutch_epa", ascending=True)
            .reset_index(drop=True)
        )
        if df_clutch.empty:
            st.info("No clutch data available for the selected filters.")
        else:
            avg_clutch = df_clutch["clutch_epa"].mean()
            fig_clutch = go.Figure(go.Bar(
                x=df_clutch["clutch_epa"],
                y=df_clutch["QB"].str.split(".").str[-1],
                orientation="h",
                marker=dict(
                    color=df_clutch["clutch_epa"],
                    colorscale=_DIVERG,
                    cmid=0,
                    showscale=False,
                ),
                customdata=list(zip(
                    df_clutch["QB"],
                    df_clutch["clutch_epa"].map(lambda v: f"{v:+.3f}"),
                    df_clutch["clutch_dropbacks"].map(lambda v: f"{int(v)}" if pd.notna(v) else "—"),
                    df_clutch["epa_per_play"].map(lambda v: f"{v:+.3f}"),
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Clutch EPA: %{customdata[1]}<br>"
                    "Clutch dropbacks: %{customdata[2]}<br>"
                    "Overall EPA: %{customdata[3]}"
                    "<extra></extra>"
                ),
                name="",
            ))
            fig_clutch.add_vline(x=0, line_dash="solid", line_color=_NEUTRAL, opacity=0.4)
            fig_clutch.add_vline(
                x=avg_clutch, line_dash="dot", line_color=_NEUTRAL, opacity=0.6,
                annotation_text=f"avg {avg_clutch:+.2f}",
                annotation_position="top right",
                annotation_font_size=9, annotation_font_color=_NEUTRAL,
            )
            _clean_fig(
                fig_clutch,
                title=dict(
                    text="<b>Clutch Factor</b>"
                         "<br><sup>EPA/dropback · Q4 · score within 8 pts (≥10 dropbacks)</sup>",
                    font_size=13, x=0,
                ),
                xaxis=dict(
                    title="Clutch EPA/dropback", tickformat="+.3f",
                    showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
                ),
                yaxis=dict(title="", showline=False, showgrid=False, zeroline=False, tickfont=dict(size=10)),
                height=max(300, len(df_clutch) * 24),
                showlegend=False,
                margin=dict(l=5, r=10, t=50, b=30),
            )
            st.plotly_chart(fig_clutch, width="stretch")

    # ── ROW 3: Context Score ───────────────────────────────────────────────────
    st.markdown("#### Context Score")
    df_ctx = (
        df_usage.dropna(subset=["context_score"])
        .sort_values("context_score", ascending=True)
        .reset_index(drop=True)
    )
    avg_ctx = df_ctx["context_score"].mean()

    fig_ctx = go.Figure(go.Bar(
        x=df_ctx["context_score"],
        y=df_ctx["QB"].str.split(".").str[-1],
        orientation="h",
        marker=dict(
            color=df_ctx["context_score"],
            colorscale="RdYlGn",
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="Score", side="right"),
                thickness=8, len=0.3, outlinewidth=0,
                x=0.99, xanchor="right",
            ),
        ),
        customdata=list(zip(
            df_ctx["QB"],
            df_ctx["context_score"].map(lambda v: f"{v:.1f}"),
            df_ctx["epa_per_play"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
            df_ctx["epa_pressure"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
            df_ctx["clutch_epa"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
            df_ctx["pass_play_weight"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"),
            df_ctx["team_rush_epa"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—"),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Context Score: %{customdata[1]}/100<br>"
            "─────────────────<br>"
            "EPA/play: %{customdata[2]}<br>"
            "EPA under pressure: %{customdata[3]}<br>"
            "Clutch EPA: %{customdata[4]}<br>"
            "Team pass%: %{customdata[5]}<br>"
            "Team rush EPA: %{customdata[6]}"
            "<extra></extra>"
        ),
        name="",
    ))
    fig_ctx.add_vline(x=50, line_dash="dot", line_color=_NEUTRAL, opacity=0.5,
                      annotation_text="median (50)", annotation_position="top right",
                      annotation_font_size=9, annotation_font_color=_NEUTRAL)
    _clean_fig(
        fig_ctx,
        title=dict(
            text="<b>Context Score</b>"
                 "<br><sup>40% EPA efficiency · 25% pressure resilience · 20% clutch · 15% pass burden — percentile-ranked 0–100</sup>",
            font_size=13, x=0,
        ),
        xaxis=dict(
            title="Context Score (0–100)", range=[0, 100],
            showline=True, linecolor="#cccccc", showgrid=False, zeroline=False,
        ),
        yaxis=dict(title="", showline=False, showgrid=False, zeroline=False, tickfont=dict(size=10)),
        height=max(300, len(df_ctx) * 24),
        showlegend=False,
        margin=dict(l=5, r=10, t=60, b=30),
    )
    st.plotly_chart(fig_ctx, width="stretch")

    # ── Data Table ────────────────────────────────────────────────────────────
    st.markdown("#### Detail")
    usage_cols = [
        "season", "QB", "Team",
        "epa_per_play", "pressure_rate", "epa_clean", "epa_pressure", "pressure_drop",
        "pass_play_weight", "team_rush_epa",
        "clutch_epa", "clutch_dropbacks",
        "context_score",
        "team_dropback_share", "snap_adj_epa",
    ]
    # keep only columns that exist (safe guard)
    usage_cols = [c for c in usage_cols if c in df_usage.columns]
    df_usage_tbl = (
        df_usage[usage_cols]
        .sort_values("context_score", ascending=False)
        .style
        .format({
            "epa_per_play":        "{:+.3f}",
            "pressure_rate":       "{:.1%}",
            "epa_clean":           "{:+.3f}",
            "epa_pressure":        "{:+.3f}",
            "pressure_drop":       "{:+.3f}",
            "pass_play_weight":    "{:.1%}",
            "team_rush_epa":       "{:+.3f}",
            "clutch_epa":          "{:+.3f}",
            "clutch_dropbacks":    "{:.0f}",
            "context_score":       "{:.1f}",
            "team_dropback_share": "{:.1%}",
            "snap_adj_epa":        "{:+.2f}",
        }, na_rep="—")
        .background_gradient(subset=["context_score"], cmap="RdYlGn", vmin=0, vmax=100)
        .background_gradient(subset=["pressure_drop"], cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        .background_gradient(subset=["clutch_epa"],    cmap="RdBu",   vmin=-0.3, vmax=0.3)
    )
    st.dataframe(df_usage_tbl, hide_index=True, width="stretch")
