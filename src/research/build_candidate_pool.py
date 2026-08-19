"""Build candidate rows with constructed positive/negative labels for historical and candidate-level ranking datasets."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from .build_labels import add_cba_next_season_labels
    from ..sources.kaggle_49leagues_source import Kaggle49LeaguesSource
    from .utils import PROCESSED_DIR, REPORTS_DIR, configure_logging, ensure_data_dirs, is_eligible_overseas_league
except ImportError:  # Allows: python src/build_candidate_pool.py
    from build_labels import add_cba_next_season_labels
    from sources.kaggle_49leagues_source import Kaggle49LeaguesSource
    from utils import PROCESSED_DIR, REPORTS_DIR, configure_logging, ensure_data_dirs, is_eligible_overseas_league


LOGGER = logging.getLogger(__name__)
EXTENDED_CBA_LABELS = PROCESSED_DIR / "cba_imports_extended.csv"

OUTPUT_COLUMNS = [
    "player_name_raw",
    "player_name_clean",
    "player_name_key",
    "season",
    "next_season",
    "league",
    "team",
    "source",
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
    "usage_proxy",
    "assist_to_turnover_ratio",
    "total_rebound_rate",
    "offensive_rebound_rate",
    "defensive_rebound_rate",
    "signed_cba_next_season",
]


def _load_kaggle_history(start_year: int | None, end_year: int | None) -> pd.DataFrame:
    source = Kaggle49LeaguesSource()
    csv_files = sorted(source.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Kaggle CSV files found in {source.data_dir}")
    dataset = source._load_dataset(csv_files)
    mapping = source._inspect_columns(dataset)
    history = source._standardise_dataset(dataset, mapping)
    history = source.normalise_history_schema(history)
    history = history[history["league"].map(is_eligible_overseas_league)].copy()
    if start_year is not None:
        history = history[pd.to_numeric(history["season_start_year"], errors="coerce") >= start_year]
    if end_year is not None:
        history = history[pd.to_numeric(history["season_start_year"], errors="coerce") <= end_year]
    return history


def build_label_distribution_summary(labelled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(labelled)
    positives = int(labelled["signed_cba_next_season"].sum()) if total else 0
    rows.append(
        {
            "summary_level": "overall",
            "group": "ALL",
            "total_candidate_rows": total,
            "positive_rows": positives,
            "negative_rows": total - positives,
            "positive_rate": round(positives / total, 6) if total else 0.0,
        }
    )

    for season, group in labelled.groupby("season", dropna=False):
        positives = int(group["signed_cba_next_season"].sum())
        rows.append(
            {
                "summary_level": "season",
                "group": season,
                "total_candidate_rows": len(group),
                "positive_rows": positives,
                "negative_rows": len(group) - positives,
                "positive_rate": round(positives / len(group), 6) if len(group) else 0.0,
            }
        )

    for league, group in labelled.groupby("league", dropna=False):
        positives = int(group["signed_cba_next_season"].sum())
        rows.append(
            {
                "summary_level": "league",
                "group": league,
                "total_candidate_rows": len(group),
                "positive_rows": positives,
                "negative_rows": len(group) - positives,
                "positive_rate": round(positives / len(group), 6) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_pool(
    cba_clean_path: Path,
    candidate_out: Path,
    labelled_out: Path,
    label_summary_out: Path,
    start_year: int | None,
    end_year: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cba_clean_path == PROCESSED_DIR / "cba_imports_clean.csv" and EXTENDED_CBA_LABELS.exists():
        LOGGER.info("Using extended CBA labels from %s", EXTENDED_CBA_LABELS)
        cba_clean_path = EXTENDED_CBA_LABELS

    if not cba_clean_path.exists():
        raise FileNotFoundError(f"Missing CBA label table: {cba_clean_path}")

    cba_labels = pd.read_csv(cba_clean_path)
    candidate_pool = _load_kaggle_history(start_year, end_year)
    candidate_pool = candidate_pool.drop_duplicates(["player_name_key", "season", "league", "team"]).copy()
    candidate_pool["next_season"] = pd.NA
    labelled = add_cba_next_season_labels(candidate_pool, cba_labels)

    for col in OUTPUT_COLUMNS:
        if col not in candidate_pool.columns:
            candidate_pool[col] = pd.NA
        if col not in labelled.columns:
            labelled[col] = pd.NA

    candidate_out.parent.mkdir(parents=True, exist_ok=True)
    labelled_out.parent.mkdir(parents=True, exist_ok=True)
    label_summary_out.parent.mkdir(parents=True, exist_ok=True)

    candidate_pool[OUTPUT_COLUMNS].drop(columns=["signed_cba_next_season"]).to_csv(candidate_out, index=False)
    labelled[OUTPUT_COLUMNS].to_csv(labelled_out, index=False)
    summary = build_label_distribution_summary(labelled)
    summary.to_csv(label_summary_out, index=False)

    LOGGER.info("Wrote eligible candidate pool to %s (%s rows).", candidate_out, len(candidate_pool))
    LOGGER.info("Wrote labelled candidate dataset to %s (%s rows).", labelled_out, len(labelled))
    LOGGER.info("Wrote label distribution summary to %s.", label_summary_out)
    return candidate_pool, labelled, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labelled candidate pool from eligible Kaggle 49-league rows.")
    parser.add_argument("--cba-clean", type=Path, default=PROCESSED_DIR / "cba_imports_clean.csv")
    parser.add_argument("--candidate-out", type=Path, default=PROCESSED_DIR / "eligible_candidate_pool.csv")
    parser.add_argument("--labelled-out", type=Path, default=PROCESSED_DIR / "labelled_candidate_dataset.csv")
    parser.add_argument("--label-summary-out", type=Path, default=REPORTS_DIR / "label_distribution_summary.csv")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    ensure_data_dirs()
    build_candidate_pool(
        args.cba_clean,
        args.candidate_out,
        args.labelled_out,
        args.label_summary_out,
        args.start_year,
        args.end_year,
    )


if __name__ == "__main__":
    main()
