"""
Module: build_features.py
Purpose: Construct ranking features, including pathway, role-volume, efficiency, and prior-CBA context features.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from .utils import (
        PROCESSED_DIR,
        cba_join_cutoff_year,
        configure_logging,
        ensure_data_dirs,
        is_chinese_cba_league,
        is_eligible_overseas_league,
    )
except ImportError:  # Allows: python src/build_features.py
    from utils import (
        PROCESSED_DIR,
        cba_join_cutoff_year,
        configure_logging,
        ensure_data_dirs,
        is_chinese_cba_league,
        is_eligible_overseas_league,
    )


LOGGER = logging.getLogger(__name__)
FEATURE_STATS = [
    "games",
    "games_started",
    "minutes",
    "minutes_per_game",
    "team_minutes_share",
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
    "two_pct",
    "three_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
    "usage_rate",
    "usage_proxy",
    "possessions_used",
    "field_goal_attempt_rate",
    "free_throw_rate",
    "turnover_rate",
    "assist_rate",
    "assist_to_turnover_ratio",
    "total_rebound_rate",
    "offensive_rebound_rate",
    "defensive_rebound_rate",
    "steal_rate",
    "block_rate",
    "points_per_36",
    "rebounds_per_36",
    "assists_per_36",
    "turnovers_per_36",
    "points_per_100_possessions",
    "assists_per_100_possessions",
    "turnovers_per_100_possessions",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
    "plus_minus",
    "pace",
    "pie",
    "data_completeness_score",
]
TREND_STATS = ["points", "minutes", "usage_rate", "usage_proxy", "ts_pct", "assist_rate", "turnover_rate", "net_rating"]


# 功能：从候选人记录中安全读取一个数值。
def _numeric_value(value: object) -> object:
    return pd.to_numeric(value, errors="coerce")


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _mean_features(history: pd.DataFrame, window: int, prefix: str) -> dict[str, object]:
    subset = history.sort_values("season_start_year", ascending=False).head(window) if "season_start_year" in history.columns else history
    features: dict[str, object] = {}
    for stat in FEATURE_STATS:
        features[f"last_{window}_{prefix}_season_avg_{stat}"] = pd.to_numeric(subset.get(stat), errors="coerce").mean()
    return features


# 功能：根据现有字段计算并添加当前所需信息。
def _add_stat_group(features: dict[str, object], history: pd.DataFrame, prefix: str) -> None:
    if history.empty:
        for stat in FEATURE_STATS:
            features[f"last_{prefix}_season_{stat}"] = pd.NA
        features.update(_mean_features(history, 2, prefix))
        features.update(_mean_features(history, 3, prefix))
        for stat in TREND_STATS:
            features[f"{prefix}_{stat}_trend_last2"] = pd.NA
        return

    last = history.iloc[0]
    for stat in FEATURE_STATS:
        features[f"last_{prefix}_season_{stat}"] = last.get(stat)
    features.update(_mean_features(history, 2, prefix))
    features.update(_mean_features(history, 3, prefix))

    last_two = history.head(2)
    if len(last_two) >= 2:
        latest = last_two.iloc[0]
        previous = last_two.iloc[1]
        for stat in TREND_STATS:
            features[f"{prefix}_{stat}_trend_last2"] = _numeric_value(latest.get(stat)) - _numeric_value(previous.get(stat))
    else:
        for stat in TREND_STATS:
            features[f"{prefix}_{stat}_trend_last2"] = pd.NA


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _prior_cba_features(prior_cba: pd.DataFrame) -> dict[str, object]:
    if prior_cba.empty:
        return {
            "has_prior_cba_experience": False,
            "prior_cba_seasons": 0,
            "prior_cba_games": 0,
            "prior_cba_minutes": 0,
            "prior_cba_points": 0,
            "prior_cba_last_season": pd.NA,
        }
    return {
        "has_prior_cba_experience": True,
        "prior_cba_seasons": prior_cba["season"].nunique(),
        "prior_cba_games": pd.to_numeric(prior_cba.get("games"), errors="coerce").sum(min_count=1),
        "prior_cba_minutes": pd.to_numeric(prior_cba.get("minutes"), errors="coerce").sum(min_count=1),
        "prior_cba_points": pd.to_numeric(prior_cba.get("points"), errors="coerce").sum(min_count=1),
        "prior_cba_last_season": prior_cba.iloc[0].get("season"),
    }


# 功能：根据现有字段计算并添加当前所需信息。
def _add_raw_history_features(features: dict[str, object], prior: pd.DataFrame) -> None:
    features["history_seasons_available"] = len(prior)
    features["number_of_overseas_leagues_played"] = prior["league"].nunique() if not prior.empty else 0
    features["number_of_sources_matched"] = prior["source"].nunique() if not prior.empty else 0

    if prior.empty:
        features.update(
            {
                "last_overseas_league": pd.NA,
                "last_overseas_team": pd.NA,
                "last_overseas_source": pd.NA,
                "last_overseas_season": pd.NA,
                "league_group": "unmatched",
                "changed_leagues_recently": pd.NA,
            }
        )
    else:
        last = prior.iloc[0]
        features.update(
            {
                "last_overseas_league": last.get("league"),
                "last_overseas_team": last.get("team"),
                "last_overseas_source": last.get("source"),
                "last_overseas_season": last.get("season"),
                "league_group": last.get("league"),
                "changed_leagues_recently": bool(prior.head(2)["league"].nunique() > 1) if len(prior) >= 2 else False,
            }
        )
    _add_stat_group(features, prior, "raw")


# 功能：根据现有字段计算并添加当前所需信息。
def _add_eligible_history_features(features: dict[str, object], eligible_prior: pd.DataFrame) -> None:
    features["eligible_history_seasons_available"] = len(eligible_prior)
    features["number_of_eligible_overseas_leagues_played"] = (
        eligible_prior["league"].nunique() if not eligible_prior.empty else 0
    )

    if eligible_prior.empty:
        features.update(
            {
                "last_eligible_overseas_league": pd.NA,
                "last_eligible_overseas_team": pd.NA,
                "last_eligible_overseas_source": pd.NA,
                "last_eligible_overseas_season": pd.NA,
                "eligible_league_group": "unmatched",
                "changed_eligible_leagues_recently": pd.NA,
            }
        )
    else:
        last = eligible_prior.iloc[0]
        features.update(
            {
                "last_eligible_overseas_league": last.get("league"),
                "last_eligible_overseas_team": last.get("team"),
                "last_eligible_overseas_source": last.get("source"),
                "last_eligible_overseas_season": last.get("season"),
                "eligible_league_group": last.get("league"),
                "changed_eligible_leagues_recently": bool(eligible_prior.head(2)["league"].nunique() > 1)
                if len(eligible_prior) >= 2
                else False,
            }
        )
    _add_stat_group(features, eligible_prior, "eligible")


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _label_features(label: pd.Series, history: pd.DataFrame) -> dict[str, object]:
    cutoff_year = cba_join_cutoff_year(label["target_cba_join_season"])
    prior = history[
        (history["player_name_key"] == label["player_name_key"])
        & (pd.to_numeric(history["season_start_year"], errors="coerce") < cutoff_year)
    ].copy()
    prior = prior.sort_values(["season_start_year", "source", "league"], ascending=[False, True, True])
    eligible_prior = prior[prior["league"].map(is_eligible_overseas_league)].copy()
    prior_cba = prior[prior["league"].map(is_chinese_cba_league)].copy()

    features: dict[str, object] = {
        "cba_season": label["cba_season"],
        "player_name_raw": label["player_name_raw"],
        "player_name_clean": label["player_name_clean"],
        "player_name_key": label["player_name_key"],
        "target_cba_join_season": label["target_cba_join_season"],
        "signed_cba_next_season": 1,
    }
    _add_raw_history_features(features, prior)
    _add_eligible_history_features(features, eligible_prior)
    features.update(_prior_cba_features(prior_cba))
    return features


# 功能：根据已有数据构建当前流程需要的结果表。
def build_features(cba_clean_path: Path, history_path: Path, output_path: Path) -> pd.DataFrame:
    if not cba_clean_path.exists():
        raise FileNotFoundError(f"Missing CBA label table: {cba_clean_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing combined player history table: {history_path}")

    labels = pd.read_csv(cba_clean_path)
    history = pd.read_csv(history_path)
    rows = [_label_features(label, history) for _, label in labels.iterrows()]
    features = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    LOGGER.info("Wrote model features to %s (%s rows).", output_path, len(features))
    return features


# 功能：执行历史特征构建并保存建模特征表。
def main() -> None:
    parser = argparse.ArgumentParser(description="Build modelling features from pre-CBA player histories.")
    parser.add_argument("--cba-clean", type=Path, default=PROCESSED_DIR / "cba_imports_clean.csv")
    parser.add_argument("--history", type=Path, default=PROCESSED_DIR / "player_history_all.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "model_features.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    ensure_data_dirs()
    build_features(args.cba_clean, args.history, args.out)


if __name__ == "__main__":
    main()
