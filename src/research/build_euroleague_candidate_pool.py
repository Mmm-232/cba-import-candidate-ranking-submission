from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from ..sources.euroleague_full_pool_source import EuroleagueFullPoolSource
    from .utils import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_data_dirs
except ImportError:
    from sources.euroleague_full_pool_source import EuroleagueFullPoolSource
    from utils import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_data_dirs


OUTPUT_COLUMNS = [
    "player_name_raw",
    "player_name_clean",
    "player_name_key",
    "season",
    "next_season",
    "league",
    "team",
    "source_id",
    "source_name",
    "source_url_or_file",
    "extraction_date",
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
    "three_pct",
    "ft_pct",
    "efg_pct",
    "ts_pct",
    "usage_proxy",
    "assist_to_turnover_ratio",
    "data_completeness_score",
    "signed_cba_next_season",
]


# 函数：_label
def _label(df: pd.DataFrame) -> pd.DataFrame:
    label_path = PROCESSED_DIR / "cba_imports_extended_verified.csv"
    if not label_path.exists():
        label_path = PROCESSED_DIR / "cba_imports_extended.csv"
    labels = pd.read_csv(label_path)
    pairs = set(zip(labels["player_name_key"].astype(str), labels["cba_season"].astype(str)))
    df = df.copy()
    df["signed_cba_next_season"] = [
        int((str(row.player_name_key), str(row.next_season)) in pairs) for row in df.itertuples(index=False)
    ]
    return df


# 函数：run
def run(start_year: int = 2020, end_year: int = 2024) -> None:
    ensure_data_dirs()
    try:
        pool = EuroleagueFullPoolSource().collect_full_pool(start_year, end_year)
        status = "success"
        error = ""
    except Exception as exc:
        pool = pd.DataFrame()
        status = "failed"
        error = str(exc)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(RAW_DIR / "euroleague_full_candidate_pool_raw.csv", index=False)
    if not pool.empty:
        pool = _label(pool)
    for col in OUTPUT_COLUMNS:
        if col not in pool.columns:
            pool[col] = pd.NA
    pool[OUTPUT_COLUMNS].to_csv(PROCESSED_DIR / "euroleague_full_candidate_pool_mapped.csv", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "status", "value": status},
            {"metric": "error", "value": error},
            {"metric": "rows", "value": len(pool)},
            {"metric": "positives", "value": int(pool["signed_cba_next_season"].sum()) if not pool.empty else 0},
            {"metric": "unique_players", "value": pool["player_name_key"].nunique() if not pool.empty else 0},
            {"metric": "seasons", "value": ", ".join(sorted(pool["season"].dropna().astype(str).unique())) if not pool.empty else ""},
        ]
    )
    summary.to_csv(REPORTS_DIR / "euroleague_full_pool_summary.csv", index=False)
    print(f"EuroLeague full pool {status}: {len(pool)} rows")


# 函数：main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    run(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
