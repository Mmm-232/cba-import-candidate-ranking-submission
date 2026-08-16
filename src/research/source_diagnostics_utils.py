from __future__ import annotations

import pandas as pd

from . import player_season_rank_utils as ps


# 函数：source_group
def source_group(row: pd.Series) -> str:
    sources = str(row.get("sources_present", row.get("source_id", ""))).lower()
    leagues = str(row.get("leagues_played_that_season", row.get("league", ""))).lower()
    source_count = len([s for s in sources.split(";") if s.strip()])
    if source_count > 1 or int(row.get("multi_league_season_flag", 0) or 0):
        return "multi_source"
    if int(row.get("has_gleague_row", 0) or 0) or "gleague" in sources or "g league" in leagues:
        return "gleague_nba_api"
    if any(term in sources or term in leagues for term in ["australian_nbl", "australian-nbl", "australian nbl"]):
        return "domestic_league_local_csv"
    if int(row.get("has_euroleague_row", 0) or 0) or int(row.get("has_eurocup_row", 0) or 0) or "euroleague" in sources or "eurocup" in leagues or "euroleague" in leagues:
        return "euroleague_api"
    if "kaggle" in sources:
        return "kaggle"
    return "other"


# 函数：add_source_group
def add_source_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_group"] = out.apply(source_group, axis=1)
    return out


# 函数：safe_rank_percentile
def safe_rank_percentile(score: pd.Series, group: pd.Series | None = None) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce")
    if group is None:
        return score.rank(method="average", pct=True).fillna(0.0)
    return score.groupby(group).rank(method="average", pct=True).fillna(0.0)


# 函数：base_rule_score
def base_rule_score(df: pd.DataFrame) -> pd.Series:
    return ps.fit_rule_score(df)


# 函数：rank_frame
def rank_frame(test: pd.DataFrame, score: pd.Series, method: str) -> pd.DataFrame:
    ranked = test.copy()
    ranked["score"] = pd.to_numeric(score, errors="coerce").fillna(-999).to_numpy()
    ranked["rank"] = ranked["score"].rank(method="first", ascending=False).astype(int)
    ranked["method"] = method
    return ranked


# 函数：evaluate_ranked
def evaluate_ranked(test: pd.DataFrame, score: pd.Series, meta: dict) -> tuple[dict, pd.DataFrame]:
    row, tp = ps.evaluate_scores(test, score, meta)
    return row, tp
