"""
Module: utils.py
Purpose: Shared utility helpers for normalisation, matching, statistics, feature helpers, and formatting.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = DATA_DIR / "reports"


# 功能：创建原始数据、处理数据、缓存和报告所需的文件夹。
def ensure_data_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# 功能：设置日志级别和显示格式，方便查看运行过程和错误。
def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# 功能：清理并统一球员姓名，同时保留姓名中的前后缀。
def normalise_player_name(raw_name: str) -> str:
    """Return a display-friendly normalised name while preserving particles/suffixes."""
    name = unicodedata.normalize("NFKC", raw_name or "").strip()
    name = name.replace("’", "'").replace("`", "'").replace("´", "'")
    name = re.sub(r"\s+", " ", name)
    return name.title()


# 功能：把球员姓名转换成便于跨数据源匹配的统一键。
def player_name_key(name: str) -> str:
    """Create a conservative matching key for joins across sources."""
    ascii_name = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", ascii_name)
    ascii_name = re.sub(r"[^a-z0-9]+", "", ascii_name)
    return ascii_name


# 功能：从赛季文本中提取开始年份。
def season_start_year(season: str) -> int:
    match = re.search(r"(19|20)\d{2}", str(season))
    if not match:
        raise ValueError(f"Cannot infer season start year from {season!r}")
    return int(match.group(0))


# 功能：把开始年份转换成 NBA 使用的赛季格式。
def nba_season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# 功能：根据 CBA 赛季计算其前一个 NBA 赛季。
def previous_nba_season(cba_season: str) -> str:
    """Map a CBA season like 2025-2026 to the previous NBA season label, e.g. 2024-25."""
    return nba_season_label(season_start_year(cba_season) - 1)


# 功能：计算目标 CBA 赛季对应的历史数据截止年份。
def cba_join_cutoff_year(cba_season: str) -> int:
    """Return the start year of the CBA season; histories must start before this."""
    return season_start_year(cba_season)


# 功能：把文本转换成可安全用于文件名或标识符的格式。
def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unknown"


# 功能：清理联赛名称，方便统一比较和筛选。
def normalise_league_name(league: object) -> str:
    text = unicodedata.normalize("NFKC", str(league or "")).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


# 功能：判断一条记录是否属于 NBA。
def is_nba_league(league: object) -> bool:
    key = normalise_league_name(league)
    return key == "nba" or key == "nationalbasketballassociation"


# 功能：判断一条记录是否属于中国 CBA 或相关国内联赛。
def is_chinese_cba_league(league: object) -> bool:
    key = normalise_league_name(league)
    if not key:
        return False
    return any(term in key for term in ["chinesecba", "chinacba", "chinese", "china", "cba"])


# 功能：判断联赛能否作为非 NBA 的海外候选经历。
def is_eligible_overseas_league(league: object) -> bool:
    if league is None or pd.isna(league):
        return False
    if is_nba_league(league):
        return False
    if is_chinese_cba_league(league):
        return False
    return True


# 功能：把指定字段安全转换成数值列。
def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


# 功能：安全执行除法，避免零分母和无效数值造成报错。
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


# 功能：根据基础统计计算真实命中率、每36分钟数据和使用率代理等指标。
def add_derived_history_metrics(history: pd.DataFrame) -> pd.DataFrame:
    """Add source-agnostic role, usage, and efficiency fields where inputs exist."""
    history = history.copy()
    for col in [
        "games",
        "minutes",
        "points",
        "rebounds",
        "assists",
        "turnovers",
        "field_goal_attempts",
        "three_point_attempts",
        "free_throw_attempts",
    ]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    games = numeric_series(history, "games")
    minutes = numeric_series(history, "minutes")
    points = numeric_series(history, "points")
    rebounds = numeric_series(history, "rebounds")
    assists = numeric_series(history, "assists")
    turnovers = numeric_series(history, "turnovers")
    fga = numeric_series(history, "field_goal_attempts")
    three_pa = numeric_series(history, "three_point_attempts")
    fta = numeric_series(history, "free_throw_attempts")

    if "minutes_per_game" not in history.columns or history["minutes_per_game"].isna().all():
        history["minutes_per_game"] = safe_divide(minutes, games)

    true_shooting_denominator = 2 * (fga + 0.44 * fta)
    if "ts_pct" not in history.columns or history["ts_pct"].isna().all():
        history["ts_pct"] = safe_divide(points, true_shooting_denominator)

    if "usage_proxy" not in history.columns or history["usage_proxy"].isna().all():
        history["usage_proxy"] = safe_divide(fga + 0.44 * fta + turnovers, minutes)

    if "field_goal_attempt_rate" not in history.columns or history["field_goal_attempt_rate"].isna().all():
        history["field_goal_attempt_rate"] = safe_divide(fga, minutes)
    if "free_throw_rate" not in history.columns or history["free_throw_rate"].isna().all():
        history["free_throw_rate"] = safe_divide(fta, fga)
    if "assist_to_turnover_ratio" not in history.columns or history["assist_to_turnover_ratio"].isna().all():
        history["assist_to_turnover_ratio"] = safe_divide(assists, turnovers)

    per36_map = {
        "points_per_36": points,
        "rebounds_per_36": rebounds,
        "assists_per_36": assists,
        "turnovers_per_36": turnovers,
    }
    for output_col, source_series in per36_map.items():
        if output_col not in history.columns or history[output_col].isna().all():
            history[output_col] = safe_divide(source_series, minutes) * 36

    return history


# 功能：计算每条球员赛季记录的字段完整度。
def add_data_completeness_score(history: pd.DataFrame, scoring_columns: list[str]) -> pd.DataFrame:
    history = history.copy()
    available_cols = [col for col in scoring_columns if col in history.columns]
    if not available_cols:
        history["data_completeness_score"] = 0.0
        return history
    history["data_completeness_score"] = history[available_cols].notna().mean(axis=1).round(3)
    return history
