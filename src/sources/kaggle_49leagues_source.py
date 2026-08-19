"""Adapter for local Kaggle 49-leagues style CSV files and column-name mapping to unified schema."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from .base import PlayerHistorySource, SourceCollectionResult

try:
    from ..utils import DATA_DIR, add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key, season_start_year
except ImportError:  # Allows direct script-style imports in some contexts.
    from utils import DATA_DIR, add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key, season_start_year


LOGGER = logging.getLogger(__name__)


SOURCE_NAME = "kaggle_49leagues"
DEFAULT_DATA_DIR = DATA_DIR / "external" / "kaggle_49leagues"

FIELD_CANDIDATES = {
    "player_name_raw": ["player", "player_name", "name"],
    "league": ["league", "competition", "competition_name"],
    "team": ["team", "team_name", "club"],
    "season": ["season", "year"],
    "games": ["gp", "games", "games_played", "g"],
    "minutes": ["min", "minutes", "minutes_played", "mp"],
    "minutes_per_game": ["min", "minutes", "minutes_per_game", "mpg", "mp"],
    "points": ["pts", "points", "points_scored", "ppg"],
    "rebounds": ["reb", "rebounds", "total_rebounds", "trb", "rpg"],
    "assists": ["ast", "assists", "apg"],
    "steals": ["stl", "steals", "spg"],
    "blocks": ["blk", "blocks", "bpg"],
    "turnovers": ["tov", "to", "turnovers"],
    "field_goals_made": ["fgm", "field_goals_made", "field_goal_made"],
    "field_goal_attempts": ["fga", "field_goal_attempts", "field_goals_attempted"],
    "three_points_made": ["3pm", "three_pm", "fg3m", "three_points_made", "three_pointers_made"],
    "three_point_attempts": ["3pa", "three_pa", "fg3a", "three_point_attempts", "three_pointers_attempted"],
    "free_throws_made": ["ftm", "free_throws_made", "free_throw_made"],
    "free_throw_attempts": ["fta", "free_throw_attempts", "free_throws_attempted"],
    "fg_pct": ["fg%", "fg_pct", "field_goal_percentage", "field_goals_percentage"],
    "three_pct": ["3p%", "3p_pct", "fg3_pct", "three_pct", "three_point_percentage"],
    "ft_pct": ["ft%", "ft_pct", "free_throw_percentage"],
}

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
    "usage_proxy",
    "field_goal_attempt_rate",
    "free_throw_rate",
    "assist_to_turnover_ratio",
    "points_per_36",
    "rebounds_per_36",
    "assists_per_36",
    "turnovers_per_36",
]


@dataclass
# 类：ColumnMapping
class ColumnMapping:
    canonical_name: str
    source_column: str | None


# 类：Kaggle49LeaguesSource
class Kaggle49LeaguesSource(PlayerHistorySource):
    name = SOURCE_NAME

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, fuzzy_threshold: int = 92) -> None:
        self.data_dir = data_dir
        self.fuzzy_threshold = fuzzy_threshold

    def collect(
        self,
        cba_labels: pd.DataFrame,
        start_year: int,
        end_year: int,
        limit: int | None = None,
    ) -> SourceCollectionResult:
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No Kaggle CSV files found in {self.data_dir}. "
                "Place the 'Basketball Players Stats per Season - 49 Leagues' CSV there first."
            )

        dataset = self._load_dataset(csv_files)
        mapping = self._inspect_columns(dataset)
        LOGGER.info("Kaggle 49 leagues column mapping: %s", {m.canonical_name: m.source_column for m in mapping})

        history_all = self._standardise_dataset(dataset, mapping)
        history_all = history_all[
            (pd.to_numeric(history_all["season_start_year"], errors="coerce") >= start_year)
            & (pd.to_numeric(history_all["season_start_year"], errors="coerce") <= end_year)
        ].copy()
        if history_all.empty:
            LOGGER.warning("Kaggle source has no rows between %s and %s.", start_year, end_year)

        players = cba_labels.drop_duplicates("player_name_key").copy()
        if limit is not None:
            players = players.head(limit)

        histories = []
        summaries = []
        source_names = history_all[["source_player_name", "source_player_name_key"]].drop_duplicates()
        choices = source_names["source_player_name"].dropna().tolist()

        for row in players.itertuples(index=False):
            exact = history_all[history_all["source_player_name_key"] == row.player_name_key].copy()
            if not exact.empty:
                matched = exact
                match_method = "exact_name_key"
                confidence = 100
                status = "matched"
                notes = ""
            else:
                match = process.extractOne(row.player_name_clean, choices, scorer=fuzz.token_sort_ratio) if choices else None
                if match and match[1] >= self.fuzzy_threshold:
                    source_name = match[0]
                    matched = history_all[history_all["source_player_name"] == source_name].copy()
                    match_method = "fuzzy_name"
                    confidence = int(match[1])
                    status = "matched"
                    notes = f"Fuzzy matched to {source_name}"
                else:
                    matched = pd.DataFrame()
                    match_method = "unmatched"
                    confidence = pd.NA
                    status = "unmatched"
                    notes = "No Kaggle 49-leagues player-season match in collected season range."

            if not matched.empty:
                matched["player_name_raw"] = row.player_name_raw
                matched["player_name_clean"] = row.player_name_clean
                matched["player_name_key"] = row.player_name_key
                matched["match_method"] = match_method
                matched["match_confidence"] = confidence
                histories.append(matched)
                completeness = matched["data_completeness_score"].mean()
                LOGGER.info("Matched %s to %s Kaggle rows via %s.", row.player_name_clean, len(matched), match_method)
            else:
                completeness = pd.NA
                LOGGER.info("Unmatched %s in Kaggle 49-leagues source.", row.player_name_clean)

            summaries.append(
                {
                    "source": self.name,
                    "player_name_raw": row.player_name_raw,
                    "player_name_clean": row.player_name_clean,
                    "player_name_key": row.player_name_key,
                    "match_status": status,
                    "match_method": match_method,
                    "match_confidence": confidence,
                    "matched_seasons": len(matched),
                    "best_match_confidence": confidence,
                    "data_completeness_score": completeness,
                    "notes": notes,
                }
            )

        history = pd.concat(histories, ignore_index=True) if histories else self.empty_history()
        summary = pd.DataFrame(summaries)
        return SourceCollectionResult(
            self.normalise_history_schema(history),
            self.normalise_summary_schema(summary),
        )

    def _load_dataset(self, csv_files: list[Path]) -> pd.DataFrame:
        frames = []
        for path in csv_files:
            LOGGER.info("Reading Kaggle CSV %s", path)
            frame = pd.read_csv(path)
            frame["_kaggle_file"] = path.name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _inspect_columns(self, dataset: pd.DataFrame) -> list[ColumnMapping]:
        normalised = {_normalise_column_name(col): col for col in dataset.columns}
        mappings: list[ColumnMapping] = []
        for canonical, candidates in FIELD_CANDIDATES.items():
            source_col = None
            for candidate in candidates:
                key = _normalise_column_name(candidate)
                if key in normalised:
                    source_col = normalised[key]
                    break
            mappings.append(ColumnMapping(canonical, source_col))

        missing_required = [
            mapping.canonical_name
            for mapping in mappings
            if mapping.source_column is None and mapping.canonical_name in {"player_name_raw", "league", "team", "season"}
        ]
        if missing_required:
            raise ValueError(
                "Kaggle CSV is missing required columns after inspection: "
                f"{missing_required}. Available columns: {list(dataset.columns)}"
            )
        return mappings

    def _standardise_dataset(self, dataset: pd.DataFrame, mappings: list[ColumnMapping]) -> pd.DataFrame:
        out = pd.DataFrame(index=dataset.index)
        for mapping in mappings:
            if mapping.source_column is None:
                out[mapping.canonical_name] = pd.NA
            else:
                out[mapping.canonical_name] = dataset[mapping.source_column]

        out["player_name_clean"] = out["player_name_raw"].map(normalise_player_name)
        out["source_player_name"] = out["player_name_clean"]
        out["source_player_name_key"] = out["player_name_clean"].map(player_name_key)
        out["player_name_key"] = out["source_player_name_key"]
        out["source_player_id"] = out["source_player_name_key"]
        out["source"] = self.name
        out["source_status"] = "ok"
        out["season_start_year"] = out["season"].map(_safe_season_start_year)

        for col in [
            "games",
            "minutes",
            "minutes_per_game",
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "field_goals_made",
            "field_goal_attempts",
            "three_points_made",
            "three_point_attempts",
            "free_throws_made",
            "free_throw_attempts",
            "fg_pct",
            "three_pct",
            "ft_pct",
        ]:
            out[col] = _coerce_numeric_or_percent(out[col])

        games = out["games"].replace(0, pd.NA)
        out["minutes_per_game"] = out["minutes"] / games
        out["fg_pct"] = out["fg_pct"].fillna(_safe_ratio(out["field_goals_made"], out["field_goal_attempts"]))
        out["three_pct"] = out["three_pct"].fillna(_safe_ratio(out["three_points_made"], out["three_point_attempts"]))
        out["ft_pct"] = out["ft_pct"].fillna(_safe_ratio(out["free_throws_made"], out["free_throw_attempts"]))

        out = add_derived_history_metrics(out)
        out = add_data_completeness_score(out, COMPLETENESS_COLUMNS)
        return out


def _normalise_column_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _safe_season_start_year(value: object) -> int | pd.NA:
    try:
        return season_start_year(str(value))
    except ValueError:
        return pd.NA


def _coerce_numeric_or_percent(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.strip()
        percent_mask = cleaned.str.endswith("%", na=False)
        numeric = pd.to_numeric(cleaned.str.replace("%", "", regex=False), errors="coerce")
        numeric.loc[percent_mask] = numeric.loc[percent_mask] / 100
        return numeric
    return pd.to_numeric(series, errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)
    return numerator / denominator
