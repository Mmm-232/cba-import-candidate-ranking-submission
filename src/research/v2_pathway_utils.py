from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import large_scale_rank_utils as lu
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs, is_chinese_cba_league, is_nba_league, season_start_year


V2_PROCESSED_DIR = PROCESSED_DIR / "v2_pathway_rebuild"
V2_REPORTS_DIR = REPORTS_DIR / "v2_pathway_rebuild"
BASE_PLAYER_SEASON = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
CBA_LABEL_CANDIDATES = [
    PROCESSED_DIR / "cba_imports_extended_verified.csv",
    PROCESSED_DIR / "cba_imports_extended.csv",
    PROCESSED_DIR / "cba_imports_clean.csv",
]

PATHWAY_TERMS = {
    "gleague": ["g league", "gleague"],
    "euroleague": ["euroleague"],
    "eurocup": ["eurocup"],
    "australian_nbl": ["australian-nbl", "australian nbl", "nbl australia"],
    "japanese_bleague": ["japanese-bleague", "japanese b.league", "b.league", "bleague", "japan"],
    "korean_kbl": ["korean-kbl", "korean kbl", "kbl", "korea"],
}


def ensure_v2_dirs() -> None:
    ensure_data_dirs()
    V2_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(BASE_PLAYER_SEASON)
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")
    if "league" not in df.columns:
        df["league"] = df.get("best_row_league", df.get("leagues_played_that_season", pd.NA))
    return df


def cba_label_path() -> Path:
    for path in CBA_LABEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No CBA label file found.")


def load_cba_labels() -> pd.DataFrame:
    labels = pd.read_csv(cba_label_path())
    labels["cba_start_year"] = labels["cba_season"].map(season_start_year)
    labels = labels[labels["player_name_key"].notna()].copy()
    return labels


def league_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(c, ""))
        for c in ["league", "leagues_played_that_season", "best_row_league", "best_underlying_league", "source_id", "sources_present"]
    ).lower()


def has_pathway(row: pd.Series, pathway: str) -> bool:
    text = league_text(row)
    return any(term in text for term in PATHWAY_TERMS[pathway])


def is_australian_nbl_row(row: pd.Series) -> bool:
    return has_pathway(row, "australian_nbl")


def eligible_overseas_mask(df: pd.DataFrame, include_nba: bool = False) -> pd.Series:
    league = df["league"] if "league" in df.columns else pd.Series(pd.NA, index=df.index)
    mask = ~league.map(is_chinese_cba_league)
    if not include_nba:
        mask = mask & ~league.map(is_nba_league)
    return mask.fillna(False)


def add_future_cba_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    cba_years = labels.groupby("player_name_key")["cba_start_year"].apply(lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
    out = df.copy()
    first_years, gaps = [], []
    within2, within3, within5, ever = [], [], [], []
    for row in out.itertuples(index=False):
        years = cba_years.get(getattr(row, "player_name_key"), [])
        start = int(getattr(row, "season_start_year")) if pd.notna(getattr(row, "season_start_year")) else None
        future = [y for y in years if start is not None and y > start]
        first = min(future) if future else pd.NA
        gap = first - start if pd.notna(first) and start is not None else pd.NA
        first_years.append(f"{int(first)}-{int(first) + 1}" if pd.notna(first) else pd.NA)
        gaps.append(gap)
        ever.append(int(pd.notna(gap)))
        within2.append(int(pd.notna(gap) and gap <= 2))
        within3.append(int(pd.notna(gap) and gap <= 3))
        within5.append(int(pd.notna(gap) and gap <= 5))
    out["signed_cba_within_2_seasons"] = within2
    out["signed_cba_within_3_seasons"] = within3
    out["signed_cba_within_5_seasons"] = within5
    out["ever_signed_cba_after_t"] = ever
    out["first_cba_season_after_t"] = first_years
    out["seasons_until_cba"] = gaps
    return out


def pool_mask(df: pd.DataFrame, pool: str) -> pd.Series:
    if pool == "broad_eligible_overseas_pool":
        return eligible_overseas_mask(df)
    if pool == "common_cba_source_league_pool":
        return eligible_overseas_mask(df) & lu.pool_mask(df, "pool_common_cba_source_leagues")
    if pool == "australian_nbl_augmented_pool":
        return eligible_overseas_mask(df) & (lu.pool_mask(df, "pool_common_cba_source_leagues") | df.apply(is_australian_nbl_row, axis=1))
    if pool == "career_pathway_signal_pool":
        flags = pd.Series(False, index=df.index)
        for pathway in PATHWAY_TERMS:
            flags = flags | df.apply(lambda r, p=pathway: has_pathway(r, p), axis=1)
        if "has_prior_cba_experience_before_t" in df.columns:
            flags = flags | pd.to_numeric(df["has_prior_cba_experience_before_t"], errors="coerce").fillna(0).gt(0)
        return eligible_overseas_mask(df) & flags
    if pool == "expanded_pathway_pool":
        # Filled later by v2_build_candidate_pools via expanded_pathway_flag.
        return eligible_overseas_mask(df) & df.get("expanded_pathway_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    raise ValueError(pool)
