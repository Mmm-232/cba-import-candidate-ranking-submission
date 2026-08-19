"""Abstract base interface for source adapters with standard fetch/parse/unify method signatures."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from ..utils import add_data_completeness_score, add_derived_history_metrics
except ImportError:  # Allows direct script-style imports in some contexts.
    from utils import add_data_completeness_score, add_derived_history_metrics


HISTORY_COLUMNS = [
    "player_name_raw",
    "player_name_clean",
    "player_name_key",
    "source",
    "league",
    "team",
    "season",
    "season_start_year",
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
    "isolation_points_per_possession",
    "isolation_frequency",
    "pick_and_roll_ball_handler_ppp",
    "pick_and_roll_roll_man_ppp",
    "post_up_ppp",
    "spot_up_ppp",
    "transition_ppp",
    "data_completeness_score",
    "match_method",
    "match_confidence",
    "source_player_id",
    "source_player_name",
    "source_status",
]

COMPLETENESS_COLUMNS = [
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
    "isolation_points_per_possession",
    "isolation_frequency",
    "pick_and_roll_ball_handler_ppp",
    "pick_and_roll_roll_man_ppp",
    "post_up_ppp",
    "spot_up_ppp",
    "transition_ppp",
]


SUMMARY_COLUMNS = [
    "source",
    "player_name_raw",
    "player_name_clean",
    "player_name_key",
    "match_status",
    "match_method",
    "match_confidence",
    "matched_seasons",
    "best_match_confidence",
    "data_completeness_score",
    "notes",
]


@dataclass
# 类：SourceCollectionResult
class SourceCollectionResult:
    history: pd.DataFrame
    summary: pd.DataFrame


# 类：PlayerHistorySource
class PlayerHistorySource(ABC):
    name: str

    @abstractmethod
    def collect(
        self,
        cba_labels: pd.DataFrame,
        start_year: int,
        end_year: int,
        limit: int | None = None,
    ) -> SourceCollectionResult:
        """Collect player-season histories for CBA label players."""

    def empty_history(self) -> pd.DataFrame:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    def empty_summary(self) -> pd.DataFrame:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    def write_raw(self, result: SourceCollectionResult, raw_path: Path) -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.normalise_history_schema(result.history).to_csv(raw_path, index=False)

    def normalise_history_schema(self, history: pd.DataFrame) -> pd.DataFrame:
        history = history.copy()
        for col in HISTORY_COLUMNS:
            if col not in history.columns:
                history[col] = pd.NA
        history = add_derived_history_metrics(history)
        history = add_data_completeness_score(history, COMPLETENESS_COLUMNS)
        return history[HISTORY_COLUMNS]

    def normalise_summary_schema(self, summary: pd.DataFrame) -> pd.DataFrame:
        summary = summary.copy()
        for col in SUMMARY_COLUMNS:
            if col not in summary.columns:
                summary[col] = pd.NA
        return summary[SUMMARY_COLUMNS]
