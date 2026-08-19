"""Generate coverage and completeness diagnostics for sources, candidate rows, and label fields."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from .utils import (
        PROCESSED_DIR,
        REPORTS_DIR,
        configure_logging,
        ensure_data_dirs,
        is_chinese_cba_league,
        is_eligible_overseas_league,
        is_nba_league,
    )
except ImportError:  # Allows: python src/data_quality_report.py
    from utils import (
        PROCESSED_DIR,
        REPORTS_DIR,
        configure_logging,
        ensure_data_dirs,
        is_chinese_cba_league,
        is_eligible_overseas_league,
        is_nba_league,
    )


LOGGER = logging.getLogger(__name__)

KEY_FIELDS = [
    "minutes",
    "points",
    "field_goal_attempts",
    "usage_proxy",
    "efg_pct",
    "ts_pct",
    "assist_to_turnover_ratio",
    "total_rebound_rate",
    "offensive_rebound_rate",
    "defensive_rebound_rate",
    "possessions_used",
]

PLAY_TYPE_FIELDS = [
    "isolation_points_per_possession",
    "isolation_frequency",
    "pick_and_roll_ball_handler_ppp",
    "pick_and_roll_roll_man_ppp",
    "post_up_ppp",
    "spot_up_ppp",
    "transition_ppp",
]


def availability(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.notna().mean()), 4)


def build_source_coverage_summary(
    cba_labels: pd.DataFrame,
    history: pd.DataFrame,
    matching_summary: pd.DataFrame,
) -> pd.DataFrame:
    total_cba_players = cba_labels["player_name_key"].nunique()
    rows: list[dict[str, object]] = []

    if history.empty:
        sources = matching_summary["source"].dropna().unique().tolist()
        if not sources:
            sources = ["unknown"]
        for source in sources:
            source_matches = matching_summary[matching_summary["source"] == source]
            matched = int((source_matches["match_status"] == "matched").sum())
            rows.append(
                {
                    "source": source,
                    "league": pd.NA,
                    "seasons_covered": 0,
                    "season_start_min": pd.NA,
                    "season_start_max": pd.NA,
                    "total_rows": 0,
                    "unique_players": 0,
                    "matched_cba_players": matched,
                    "unmatched_cba_players": max(total_cba_players - matched, 0),
                    "match_rate": round(matched / total_cba_players, 4) if total_cba_players else 0.0,
                    "average_data_completeness_score": pd.NA,
                }
            )
        return pd.DataFrame(rows)

    for (source, league), group in history.groupby(["source", "league"], dropna=False):
        source_matches = matching_summary[matching_summary["source"] == source]
        matched_players_for_source = set(source_matches.loc[source_matches["match_status"] == "matched", "player_name_key"])
        matched_in_league = group.loc[group["player_name_key"].isin(matched_players_for_source), "player_name_key"].nunique()
        rows.append(
            {
                "source": source,
                "league": league,
                "seasons_covered": group["season"].nunique(),
                "season_start_min": pd.to_numeric(group["season_start_year"], errors="coerce").min(),
                "season_start_max": pd.to_numeric(group["season_start_year"], errors="coerce").max(),
                "total_rows": len(group),
                "unique_players": group["player_name_key"].nunique(),
                "matched_cba_players": matched_in_league,
                "unmatched_cba_players": max(total_cba_players - matched_in_league, 0),
                "match_rate": round(matched_in_league / total_cba_players, 4) if total_cba_players else 0.0,
                "average_data_completeness_score": pd.to_numeric(
                    group.get("data_completeness_score"), errors="coerce"
                ).mean(),
            }
        )

    source_totals = []
    for source, group in history.groupby("source", dropna=False):
        source_matches = matching_summary[matching_summary["source"] == source]
        matched = int((source_matches["match_status"] == "matched").sum())
        source_totals.append(
            {
                "source": source,
                "league": "ALL",
                "seasons_covered": group["season"].nunique(),
                "season_start_min": pd.to_numeric(group["season_start_year"], errors="coerce").min(),
                "season_start_max": pd.to_numeric(group["season_start_year"], errors="coerce").max(),
                "total_rows": len(group),
                "unique_players": group["player_name_key"].nunique(),
                "matched_cba_players": matched,
                "unmatched_cba_players": max(total_cba_players - matched, 0),
                "match_rate": round(matched / total_cba_players, 4) if total_cba_players else 0.0,
                "average_data_completeness_score": pd.to_numeric(
                    group.get("data_completeness_score"), errors="coerce"
                ).mean(),
            }
        )

    return pd.DataFrame(source_totals + rows)


def build_field_completeness_summary(history: pd.DataFrame) -> pd.DataFrame:
    fields = KEY_FIELDS + PLAY_TYPE_FIELDS
    rows: list[dict[str, object]] = []
    if history.empty:
        for field in fields:
            rows.append(
                {
                    "source": pd.NA,
                    "league": pd.NA,
                    "field": field,
                    "field_group": "play_type" if field in PLAY_TYPE_FIELDS else "key",
                    "total_rows": 0,
                    "non_null_rows": 0,
                    "availability_pct": 0.0,
                }
            )
        return pd.DataFrame(rows)

    groups = [(("ALL", "ALL"), history)]
    groups.extend(list(history.groupby(["source", "league"], dropna=False)))
    groups.extend([((source, "ALL"), group) for source, group in history.groupby("source", dropna=False)])

    for (source, league), group in groups:
        for field in fields:
            if field in group.columns:
                non_null = int(group[field].notna().sum())
                pct = availability(group[field])
            else:
                non_null = 0
                pct = 0.0
            rows.append(
                {
                    "source": source,
                    "league": league,
                    "field": field,
                    "field_group": "play_type" if field in PLAY_TYPE_FIELDS else "key",
                    "total_rows": len(group),
                    "non_null_rows": non_null,
                    "availability_pct": pct,
                }
            )

    return pd.DataFrame(rows)


def build_league_filter_summary(cba_labels: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    total_players = cba_labels["player_name_key"].nunique()
    if history.empty:
        return pd.DataFrame(
            [
                {
                    "total_history_rows": 0,
                    "eligible_overseas_rows": 0,
                    "excluded_nba_rows": 0,
                    "excluded_chinese_cba_rows": 0,
                    "excluded_other_rows": 0,
                    "matched_players_with_eligible_overseas_history": 0,
                    "matched_players_with_prior_cba_experience_only": 0,
                    "matched_players_with_any_history": 0,
                    "unmatched_players_after_filtering": total_players,
                }
            ]
        )

    nba_mask = history["league"].map(is_nba_league)
    cba_mask = history["league"].map(is_chinese_cba_league)
    eligible_mask = history["league"].map(is_eligible_overseas_league)
    excluded_other_mask = ~(nba_mask | cba_mask | eligible_mask)

    eligible_players = set(history.loc[eligible_mask, "player_name_key"].dropna())
    prior_cba_players = set(history.loc[cba_mask, "player_name_key"].dropna())
    any_history_players = set(history["player_name_key"].dropna())
    prior_cba_only_players = prior_cba_players - eligible_players

    return pd.DataFrame(
        [
            {
                "total_history_rows": len(history),
                "eligible_overseas_rows": int(eligible_mask.sum()),
                "excluded_nba_rows": int(nba_mask.sum()),
                "excluded_chinese_cba_rows": int(cba_mask.sum()),
                "excluded_other_rows": int(excluded_other_mask.sum()),
                "matched_players_with_eligible_overseas_history": len(eligible_players),
                "matched_players_with_prior_cba_experience_only": len(prior_cba_only_players),
                "matched_players_with_any_history": len(any_history_players),
                "unmatched_players_after_filtering": total_players - len(eligible_players),
            }
        ]
    )


def write_reports(
    cba_clean_path: Path,
    history_path: Path,
    matching_summary_path: Path,
    source_coverage_out: Path,
    field_completeness_out: Path,
    league_filter_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (cba_clean_path, history_path, matching_summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    cba_labels = pd.read_csv(cba_clean_path)
    history = pd.read_csv(history_path)
    matching_summary = pd.read_csv(matching_summary_path)

    source_coverage = build_source_coverage_summary(cba_labels, history, matching_summary)
    field_completeness = build_field_completeness_summary(history)
    league_filter = build_league_filter_summary(cba_labels, history)

    source_coverage_out.parent.mkdir(parents=True, exist_ok=True)
    field_completeness_out.parent.mkdir(parents=True, exist_ok=True)
    source_coverage.to_csv(source_coverage_out, index=False)
    field_completeness.to_csv(field_completeness_out, index=False)
    league_filter.to_csv(league_filter_out, index=False)

    LOGGER.info("Wrote source coverage summary to %s", source_coverage_out)
    LOGGER.info("Wrote field completeness summary to %s", field_completeness_out)
    LOGGER.info("Wrote league filter summary to %s", league_filter_out)
    return source_coverage, field_completeness, league_filter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source coverage and field completeness reports.")
    parser.add_argument("--cba-clean", type=Path, default=PROCESSED_DIR / "cba_imports_clean.csv")
    parser.add_argument("--history", type=Path, default=PROCESSED_DIR / "player_history_all.csv")
    parser.add_argument("--matching-summary", type=Path, default=REPORTS_DIR / "matching_summary.csv")
    parser.add_argument("--source-coverage-out", type=Path, default=REPORTS_DIR / "source_coverage_summary.csv")
    parser.add_argument("--field-completeness-out", type=Path, default=REPORTS_DIR / "field_completeness_summary.csv")
    parser.add_argument("--league-filter-out", type=Path, default=REPORTS_DIR / "league_filter_summary.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    ensure_data_dirs()
    write_reports(
        args.cba_clean,
        args.history,
        args.matching_summary,
        args.source_coverage_out,
        args.field_completeness_out,
        args.league_filter_out,
    )


if __name__ == "__main__":
    main()
