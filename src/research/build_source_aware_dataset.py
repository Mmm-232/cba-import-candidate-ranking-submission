from __future__ import annotations

import argparse

import pandas as pd

try:
    from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs
except ImportError:
    from utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


def _source_group(source_id: object) -> str:
    text = str(source_id or "").lower()
    if "kaggle" in text:
        return "historical_kaggle"
    if "euroleague" in text:
        return "euroleague_api"
    if "gleague" in text:
        return "gleague"
    return "other"


def run() -> None:
    ensure_data_dirs()
    df = pd.read_csv(PROCESSED_DIR / "labelled_candidate_dataset_multisource.csv")
    if "source_id" not in df.columns:
        df["source_id"] = df.get("source", "unknown")
    df["source_group"] = df["source_id"].map(_source_group)
    df["is_kaggle_source"] = df["source_group"].eq("historical_kaggle").astype(int)
    df["is_euroleague_api_source"] = df["source_group"].eq("euroleague_api").astype(int)
    df["is_gleague_source"] = df["source_group"].eq("gleague").astype(int)
    df["is_2020_or_later"] = (pd.to_numeric(df["season_start_year"], errors="coerce") >= 2020).astype(int)
    df["seasons_since_2020"] = (pd.to_numeric(df["season_start_year"], errors="coerce") - 2020).clip(lower=0)
    df["is_recent_source"] = df["is_2020_or_later"]
    df["is_historical_source"] = 1 - df["is_recent_source"]
    # Placeholders are overwritten inside time-based training folds.
    df["source_positive_rate_train_only"] = 0.0
    df["league_positive_rate_train_only"] = 0.0
    df.to_csv(PROCESSED_DIR / "labelled_candidate_dataset_source_aware.csv", index=False)

    summary = []
    for level in ["source_id", "source_group", "season", "league"]:
        part = df.groupby(level, dropna=False).agg(rows=("player_name_key", "size"), positives=("signed_cba_next_season", "sum")).reset_index()
        part["positive_rate"] = part["positives"] / part["rows"]
        part.insert(0, "summary_level", level)
        part = part.rename(columns={level: "group"})
        summary.append(part)
    pd.concat(summary, ignore_index=True).to_csv(REPORTS_DIR / "source_aware_dataset_summary.csv", index=False)
    print(f"Wrote source-aware dataset: {len(df)} rows, {int(df['signed_cba_next_season'].sum())} positives")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
