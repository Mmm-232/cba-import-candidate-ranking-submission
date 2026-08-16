"""
Module: gleague_nba_api_source.py
Purpose: G League source adapter using nba_api with per-game-to-total scaling normalization.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

try:
    from ..research.audit_gleague_nba_api import SEASONS, _fetch
    from ..utils import add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key
except ImportError:
    from audit_gleague_nba_api import SEASONS, _fetch
    from utils import add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key


COMPLETENESS_COLUMNS = [
    "games",
    "minutes",
    "minutes_per_game",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
    "usage_rate",
    "usage_proxy",
    "assist_to_turnover_ratio",
]


# 功能：把 NBA API 赛季格式转换成项目赛季格式。
def _season_project_label(nba_season: str) -> str:
    start = int(str(nba_season)[:4])
    return f"{start}-{start + 1}"


# 功能：根据当前赛季生成下一赛季标签。
def _next_project_label(nba_season: str) -> str:
    start = int(str(nba_season)[:4])
    return f"{start + 1}-{start + 2}"


# 类：GLeagueNbaApiSource
class GLeagueNbaApiSource:
    source_id = "gleague_nba_api"
    source_name = "NBA Stats API G League via nba_api"
    source_url_or_file = "nba_api.stats.endpoints.leaguedashplayerstats?LeagueID=20"

    # 功能：保存初始化参数，并准备当前数据源需要的目录和配置。
    def __init__(self, seasons: list[str] | None = None) -> None:
        self.seasons = seasons or SEASONS

    # 功能：从 G League 接口读取未经统一映射的原始统计。
    def collect_raw(self) -> pd.DataFrame:
        frames = []
        for season in self.seasons:
            for measure_type in ["Base", "Advanced"]:
                try:
                    df = _fetch(season, measure_type)
                except Exception as exc:  # noqa: BLE001
                    frames.append(pd.DataFrame([{"api_season": season, "measure_type": measure_type, "source_error": str(exc)}]))
                    continue
                df = df.copy()
                df["api_season"] = season
                df["measure_type"] = measure_type
                df["league_id"] = "20"
                frames.append(df)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    # 功能：把 G League 原始统计转换成统一候选人结构。
    def collect_mapped(self) -> pd.DataFrame:
        mapped = []
        for season in self.seasons:
            try:
                basic_raw = _fetch(season, "Base")
            except Exception:
                continue
            if basic_raw.empty or "PLAYER_NAME" not in basic_raw.columns:
                continue
            basic = self._map_basic(basic_raw, season)
            try:
                advanced_raw = _fetch(season, "Advanced")
                advanced = self._map_advanced(advanced_raw)
                if not advanced.empty:
                    basic = basic.merge(advanced, on="source_player_id", how="left")
            except Exception:
                pass
            mapped.append(basic)

        if not mapped:
            return pd.DataFrame()

        out = pd.concat(mapped, ignore_index=True)
        out = add_derived_history_metrics(out)
        out = add_data_completeness_score(out, COMPLETENESS_COLUMNS)
        return out

    # 功能：把 G League 每场基础统计转换为赛季总量字段。
    def _map_basic(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        gp = pd.to_numeric(df.get("GP"), errors="coerce")
        min_pg = pd.to_numeric(df.get("MIN"), errors="coerce")

        # 功能：用每场数乘以出场次数，得到正确的赛季总量。
        def total(col: str) -> pd.Series:
            return pd.to_numeric(df.get(col), errors="coerce") * gp

        out["player_name_raw"] = df.get("PLAYER_NAME")
        out["player_name_clean"] = out["player_name_raw"].map(normalise_player_name)
        out["player_name_key"] = out["player_name_clean"].map(player_name_key)
        out["season"] = _season_project_label(season)
        out["next_season"] = _next_project_label(season)
        out["league"] = "G League"
        out["team"] = df.get("TEAM_ABBREVIATION")
        out["source_id"] = self.source_id
        out["source_name"] = self.source_name
        out["source_url_or_file"] = self.source_url_or_file
        out["extraction_date"] = date.today().isoformat()
        out["source_confidence"] = "nba_stats_api_league_id_20"
        out["verification_status"] = "source_extracted"
        out["stat_scale_source"] = "pergame_converted_to_totals"
        out["games"] = df.get("GP")
        out["minutes_per_game"] = min_pg
        out["points_per_game"] = df.get("PTS")
        out["rebounds_per_game"] = df.get("REB")
        out["assists_per_game"] = df.get("AST")
        out["minutes"] = min_pg * gp
        out["points"] = total("PTS")
        out["rebounds"] = total("REB")
        out["assists"] = total("AST")
        out["steals"] = total("STL")
        out["blocks"] = total("BLK")
        out["turnovers"] = total("TOV")
        out["field_goal_attempts"] = total("FGA")
        out["three_point_attempts"] = total("FG3A")
        out["free_throw_attempts"] = total("FTA")
        out["fg_pct"] = df.get("FG_PCT")
        out["three_pct"] = df.get("FG3_PCT")
        out["ft_pct"] = df.get("FT_PCT")
        out["source_player_id"] = df.get("PLAYER_ID").astype(str) if "PLAYER_ID" in df.columns else pd.NA
        out["source_team_id"] = df.get("TEAM_ID")
        return out

    # 功能：映射 G League 可用的高级统计字段。
    def _map_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "PLAYER_ID" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame(index=df.index)
        out["source_player_id"] = df.get("PLAYER_ID").astype(str)
        out["efg_pct"] = df.get("EFG_PCT")
        out["ts_pct"] = df.get("TS_PCT")
        out["usage_rate"] = df.get("USG_PCT")
        out["assist_to_turnover_ratio"] = df.get("AST_TO")
        out["assist_rate"] = df.get("AST_PCT")
        out["turnover_rate"] = df.get("TM_TOV_PCT")
        out["total_rebound_rate"] = df.get("REB_PCT")
        out["offensive_rebound_rate"] = df.get("OREB_PCT")
        out["defensive_rebound_rate"] = df.get("DREB_PCT")
        out["offensive_rating"] = df.get("OFF_RATING")
        out["defensive_rating"] = df.get("DEF_RATING")
        out["net_rating"] = df.get("NET_RATING")
        out["possessions_used"] = df.get("POSS")
        out["advanced_stat_scale_source"] = "advanced_official"
        return out
