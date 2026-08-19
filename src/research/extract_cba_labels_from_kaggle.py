from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from ..sources.kaggle_49leagues_source import Kaggle49LeaguesSource
    from .utils import DATA_DIR, REPORTS_DIR, configure_logging, ensure_data_dirs, is_chinese_cba_league, normalise_player_name, player_name_key, season_start_year
except ImportError:  # Allows: python src/extract_cba_labels_from_kaggle.py
    from sources.kaggle_49leagues_source import Kaggle49LeaguesSource
    from utils import DATA_DIR, REPORTS_DIR, configure_logging, ensure_data_dirs, is_chinese_cba_league, normalise_player_name, player_name_key, season_start_year


LOGGER = logging.getLogger(__name__)
MANUAL_DIR = DATA_DIR / "manual"
DEFAULT_OUT = MANUAL_DIR / "cba_imports_from_kaggle_2015_2020.csv"
DEFAULT_REPORT = REPORTS_DIR / "cba_label_coverage_summary.csv"
START_YEAR = 2015
END_YEAR = 2019


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalised = {"".join(ch.lower() for ch in col if ch.isalnum()): col for col in columns}
    for candidate in candidates:
        key = "".join(ch.lower() for ch in candidate if ch.isalnum())
        if key in normalised:
            return normalised[key]
    return None


def _is_domestic_chinese(nationality: object) -> bool:
    value = str(nationality or "").strip().lower()
    if not value or value == "nan":
        return False
    return value in {"china", "chinese", "chn", "china/chinese"}


def _to_cba_season(season: object) -> str:
    start = season_start_year(str(season))
    return f"{start}-{start + 1}"


def extract_labels(output_path: Path, report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Kaggle49LeaguesSource()
    csv_files = sorted(source.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Kaggle CSV files found in {source.data_dir}")

    dataset = source._load_dataset(csv_files)
    mapping = source._inspect_columns(dataset)
    mapped = {item.canonical_name: item.source_column for item in mapping}
    nationality_col = _find_column(list(dataset.columns), ["nationality", "country", "nation"])

    required = ["player_name_raw", "league", "team", "season"]
    missing = [col for col in required if mapped.get(col) is None]
    if missing:
        raise ValueError(f"Cannot extract CBA labels; missing mapped columns: {missing}")

    working = pd.DataFrame(
        {
            "player_name_raw": dataset[mapped["player_name_raw"]],
            "team": dataset[mapped["team"]],
            "league": dataset[mapped["league"]],
            "season_raw": dataset[mapped["season"]],
        }
    )
    working["nationality"] = dataset[nationality_col] if nationality_col else pd.NA
    working["season_start_year"] = working["season_raw"].map(lambda value: season_start_year(str(value)))

    cba_rows = working[working["league"].map(is_chinese_cba_league)].copy()
    cba_rows = cba_rows[(cba_rows["season_start_year"] >= START_YEAR) & (cba_rows["season_start_year"] <= END_YEAR)]

    has_nationality = nationality_col is not None
    domestic_mask = cba_rows["nationality"].map(_is_domestic_chinese) if has_nationality else pd.Series(False, index=cba_rows.index)
    excluded_domestic = int(domestic_mask.sum())
    kept = cba_rows[~domestic_mask].copy()
    kept["cba_season"] = kept["season_raw"].map(_to_cba_season)
    kept["player_name_clean"] = kept["player_name_raw"].map(normalise_player_name)
    kept["player_name_key"] = kept["player_name_clean"].map(player_name_key)
    kept["source"] = "kaggle_49leagues"
    kept["source_note"] = "Extracted from Kaggle 49-league Chinese-CBA rows; use for historical CBA labels only."
    kept["verification_status"] = "auto_extracted" if has_nationality else "needs_manual_review"

    output_cols = [
        "cba_season",
        "player_name_raw",
        "player_name_clean",
        "player_name_key",
        "team",
        "league",
        "nationality",
        "source",
        "source_note",
        "verification_status",
    ]
    labels = kept[output_cols].drop_duplicates(["cba_season", "player_name_key", "team"]).sort_values(
        ["cba_season", "player_name_clean", "team"]
    )

    report = _build_report(labels, excluded_domestic, has_nationality)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)
    report.to_csv(report_path, index=False)
    LOGGER.info("Wrote Kaggle-derived CBA labels to %s (%s rows).", output_path, len(labels))
    LOGGER.info("Wrote CBA label coverage summary to %s.", report_path)
    return labels, report


def _build_report(labels: pd.DataFrame, excluded_domestic: int, has_nationality: bool) -> pd.DataFrame:
    rows = []
    for season, group in labels.groupby("cba_season", dropna=False):
        rows.append(
            {
                "season": season,
                "records": len(group),
                "unique_players": group["player_name_key"].nunique(),
                "auto_extracted_rows": int((group["verification_status"] == "auto_extracted").sum()),
                "needs_manual_review_rows": int((group["verification_status"] == "needs_manual_review").sum()),
                "excluded_domestic_players": pd.NA,
                "nationality_available": has_nationality,
            }
        )
    rows.append(
        {
            "season": "ALL",
            "records": len(labels),
            "unique_players": labels["player_name_key"].nunique(),
            "auto_extracted_rows": int((labels["verification_status"] == "auto_extracted").sum()),
            "needs_manual_review_rows": int((labels["verification_status"] == "needs_manual_review").sum()),
            "excluded_domestic_players": excluded_domestic,
            "nationality_available": has_nationality,
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract historical CBA import label candidates from Kaggle Chinese-CBA rows.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    ensure_data_dirs()
    extract_labels(args.out, args.report_out)


if __name__ == "__main__":
    main()
