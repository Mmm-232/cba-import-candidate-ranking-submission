from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from euroleague import EuroleagueClient
from rapidfuzz import fuzz, process

from .base import PlayerHistorySource, SourceCollectionResult

try:
    from ..utils import (
        CACHE_DIR,
        add_data_completeness_score,
        add_derived_history_metrics,
        normalise_player_name,
        player_name_key,
    )
except ImportError:  # Allows direct script-style imports in some contexts.
    from utils import CACHE_DIR, add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key


LOGGER = logging.getLogger(__name__)


COMPETITIONS = {
    "E": "EuroLeague",
    "U": "EuroCup",
}

EUROLEAGUE_COMPLETENESS_COLUMNS = [
    "games",
    "games_started",
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
    "two_pct",
    "three_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
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
    "points_per_36",
    "rebounds_per_36",
    "assists_per_36",
    "turnovers_per_36",
    "data_completeness_score",
]


@dataclass
# 类：EuroleagueApiClient
# 类：EuroleagueApiClient
class EuroleagueApiClient:
    cache_dir: Path = CACHE_DIR / "euroleague"
    delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at = 0.0

    def fetch_player_stats(self, competition_code: str, season_code: str, stat_kind: str) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{competition_code}_{season_code}_{stat_kind}_per_game.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

        LOGGER.info("Requesting %s %s %s player stats", COMPETITIONS[competition_code], season_code, stat_kind)
        with EuroleagueClient() as client:
            endpoint = getattr(client.v3.player_stats, stat_kind)
            data = endpoint(
                competition_code,
                season_code=season_code,
                season_mode="Single",
                statistic_mode="perGame",
                limit=1000,
            )
        self._last_request_at = time.monotonic()
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data


# 类：EuroleagueSource
# 类：EuroleagueSource
class EuroleagueSource(PlayerHistorySource):
    name = "euroleague"

    def __init__(self, fuzzy_threshold: int = 90) -> None:
        self.client = EuroleagueApiClient()
        self.fuzzy_threshold = fuzzy_threshold

    def collect(
        self,
        cba_labels: pd.DataFrame,
        start_year: int,
        end_year: int,
        limit: int | None = None,
    ) -> SourceCollectionResult:
        players = cba_labels.drop_duplicates("player_name_key").copy()
        if limit is not None:
            players = players.head(limit)

        LOGGER.info(
            "Collecting EuroLeague/EuroCup histories for %s unique players across %s-%s.",
            len(players),
            start_year,
            end_year,
        )
        season_frames = []
        for year in range(start_year, end_year + 1):
            for competition_code in COMPETITIONS:
                season_code = f"{competition_code}{year}"
                try:
                    season_frames.append(self._fetch_competition_season(competition_code, season_code, year))
                except Exception as exc:
                    LOGGER.warning("Euroleague source failed for %s: %s", season_code, exc)

        if not season_frames:
            return SourceCollectionResult(self.empty_history(), self._summary_no_source_data(players))

        all_stats = pd.concat(season_frames, ignore_index=True)
        all_stats = add_derived_history_metrics(all_stats)
        all_stats = add_data_completeness_score(all_stats, EUROLEAGUE_COMPLETENESS_COLUMNS)

        histories = []
        summaries = []
        source_names = all_stats[["source_player_name", "source_player_name_key"]].drop_duplicates()
        choices = source_names["source_player_name"].tolist()

        for row in players.itertuples(index=False):
            exact = all_stats[all_stats["source_player_name_key"] == row.player_name_key].copy()
            if not exact.empty:
                matched = exact
                match_method = "exact_name_key"
                confidence = 100
                status = "matched"
                notes = ""
            else:
                match = process.extractOne(row.player_name_clean, choices, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= self.fuzzy_threshold:
                    source_name = match[0]
                    matched = all_stats[all_stats["source_player_name"] == source_name].copy()
                    match_method = "fuzzy_name"
                    confidence = int(match[1])
                    status = "matched"
                    notes = f"Fuzzy matched to {source_name}"
                else:
                    matched = pd.DataFrame()
                    match_method = "unmatched"
                    confidence = pd.NA
                    status = "unmatched"
                    notes = "No EuroLeague/EuroCup player-season match in collected season range."

            if not matched.empty:
                matched["player_name_raw"] = row.player_name_raw
                matched["player_name_clean"] = row.player_name_clean
                matched["player_name_key"] = row.player_name_key
                matched["match_method"] = match_method
                matched["match_confidence"] = confidence
                histories.append(matched)
                LOGGER.info(
                    "Matched %s to %s EuroLeague/EuroCup season rows via %s.",
                    row.player_name_clean,
                    len(matched),
                    match_method,
                )
                completeness = matched["data_completeness_score"].mean()
            else:
                LOGGER.info("Unmatched %s in EuroLeague/EuroCup source.", row.player_name_clean)
                completeness = pd.NA

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

    def _fetch_competition_season(self, competition_code: str, season_code: str, season_start_year: int) -> pd.DataFrame:
        traditional_raw = self.client.fetch_player_stats(competition_code, season_code, "traditional")
        traditional = self._standardise_traditional(
            traditional_raw.get("players", []),
            competition_code,
            season_code,
            season_start_year,
        )

        try:
            advanced_raw = self.client.fetch_player_stats(competition_code, season_code, "advanced")
            advanced = self._standardise_advanced(advanced_raw.get("players", []))
            return traditional.merge(advanced, on="source_player_id", how="left")
        except Exception as exc:
            LOGGER.warning("Euroleague advanced stats unavailable for %s; keeping basic stats only: %s", season_code, exc)
            return traditional

    def _standardise_traditional(
        self,
        players: list[dict[str, Any]],
        competition_code: str,
        season_code: str,
        season_start_year: int,
    ) -> pd.DataFrame:
        rows = []
        for item in players:
            player = item.get("player", {}) or {}
            team = player.get("team", {}) or {}
            source_name = _normalise_euroleague_player_name(player.get("name"))
            rows.append(
                {
                    "source": self.name,
                    "league": COMPETITIONS[competition_code],
                    "team": team.get("name"),
                    "season": season_code,
                    "season_start_year": season_start_year,
                    "games": item.get("gamesPlayed"),
                    "games_started": item.get("gamesStarted"),
                    "minutes": item.get("minutesPlayed"),
                    "minutes_per_game": item.get("minutesPlayed"),
                    "points": item.get("pointsScored"),
                    "rebounds": item.get("totalRebounds"),
                    "assists": item.get("assists"),
                    "steals": item.get("steals"),
                    "blocks": item.get("blocks"),
                    "turnovers": item.get("turnovers"),
                    "field_goal_attempts": _sum_values(item.get("twoPointersAttempted"), item.get("threePointersAttempted")),
                    "three_point_attempts": item.get("threePointersAttempted"),
                    "free_throw_attempts": item.get("freeThrowsAttempted"),
                    "fg_pct": _calculate_fg_pct(item),
                    "two_pct": _parse_percent(item.get("twoPointersPercentage")),
                    "three_pct": _parse_percent(item.get("threePointersPercentage")),
                    "ft_pct": _parse_percent(item.get("freeThrowsPercentage")),
                    "source_player_id": player.get("code"),
                    "source_player_name": source_name,
                    "source_status": "ok",
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["source_player_name_key"] = out["source_player_name"].map(player_name_key)
        return out

    def _standardise_advanced(self, players: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for item in players:
            player = item.get("player", {}) or {}
            rows.append(
                {
                    "source_player_id": player.get("code"),
                    "efg_pct": _parse_percent(item.get("effectiveFieldGoalPercentage")),
                    "ts_pct": _parse_percent(item.get("trueShootingPercentage")),
                    "offensive_rebound_rate": _parse_percent(item.get("offensiveReboundsPercentage")),
                    "defensive_rebound_rate": _parse_percent(item.get("defensiveReboundsPercentage")),
                    "total_rebound_rate": _parse_percent(item.get("reboundsPercentage")),
                    "assist_to_turnover_ratio": item.get("assistsToTurnoversRatio"),
                    "assist_rate": _parse_percent(item.get("assistsRatio")),
                    "turnover_rate": _parse_percent(item.get("turnoversRatio")),
                    "free_throw_rate": _parse_percent(item.get("freeThrowsRate")),
                    "possessions_used": item.get("possesions"),
                }
            )
        return pd.DataFrame(rows)

    def _summary_no_source_data(self, players: pd.DataFrame) -> pd.DataFrame:
        summary = pd.DataFrame(
            {
                "source": self.name,
                "player_name_raw": players["player_name_raw"],
                "player_name_clean": players["player_name_clean"],
                "player_name_key": players["player_name_key"],
                "match_status": "source_error",
                "match_method": "source_error",
                "match_confidence": pd.NA,
                "matched_seasons": 0,
                "best_match_confidence": pd.NA,
                "data_completeness_score": pd.NA,
                "notes": "No EuroLeague/EuroCup source season data was collected.",
            }
        )
        return self.normalise_summary_schema(summary)


def _parse_percent(value: Any) -> float | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if not value:
            return pd.NA
    try:
        return float(value) / 100
    except (TypeError, ValueError):
        return pd.NA


def _sum_values(*values: Any) -> float | pd.NA:
    numeric = [pd.to_numeric(value, errors="coerce") for value in values]
    numeric = [value for value in numeric if not pd.isna(value)]
    if not numeric:
        return pd.NA
    return float(sum(numeric))


def _calculate_fg_pct(item: dict[str, Any]) -> float | pd.NA:
    made = _sum_values(item.get("twoPointersMade"), item.get("threePointersMade"))
    attempted = _sum_values(item.get("twoPointersAttempted"), item.get("threePointersAttempted"))
    if pd.isna(made) or pd.isna(attempted) or attempted == 0:
        return pd.NA
    return float(made) / float(attempted)


def _normalise_euroleague_player_name(raw_name: str | None) -> str:
    if not raw_name:
        return ""
    name = raw_name.strip()
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        name = f"{first} {last}"
    return normalise_player_name(name)
