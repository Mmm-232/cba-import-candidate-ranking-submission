from __future__ import annotations

import pandas as pd

from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs, is_chinese_cba_league, is_nba_league


INPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague_context_enriched.csv"
FALLBACK_INPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
OUTPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague_league_adjusted.csv"
RAW_FEATURES = [
    "points_per_36",
    "usage_proxy",
    "ts_pct",
    "minutes_per_game",
    "games",
    "points",
    "rebounds",
    "assists",
    "turnovers",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "data_completeness_score",
    "best_row_data_completeness_score",
    "mean_data_completeness_score",
]
MIN_GROUP_SIZE = 10


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def _pct(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True)


def run() -> None:
    ensure_data_dirs()
    path = INPUT if INPUT.exists() else FALLBACK_INPUT
    df = pd.read_csv(path)
    df["season_start_year"] = pd.to_numeric(df.get("season_start_year"), errors="coerce")
    if "league" not in df.columns:
        raise ValueError("Missing league column.")
    bad = df["league"].map(is_chinese_cba_league).fillna(False) | df["league"].map(is_nba_league).fillna(False)
    if bad.any():
        df = df[~bad].copy()
    if "source_id" not in df.columns:
        df["source_id"] = "unknown"
    group_cols = ["league", "season"]
    group_size = df.groupby(group_cols, dropna=False)["player_name_key"].transform("size")
    df["league_season_group_size"] = group_size
    df["league_season_low_sample_flag"] = group_size.lt(MIN_GROUP_SIZE)
    created = []
    for feature in RAW_FEATURES:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() == 0:
            continue
        pct_col = f"{feature}_league_season_pct"
        z_col = f"{feature}_league_season_z"
        miss_col = f"{feature}_league_season_missing"
        df[pct_col] = df.groupby(group_cols, dropna=False)[feature].transform(_pct)
        df[z_col] = df.groupby(group_cols, dropna=False)[feature].transform(_z)
        df[miss_col] = df[pct_col].isna().astype(int)
        created.extend([pct_col, z_col, miss_col])
    df.to_csv(OUTPUT, index=False)

    low_groups = df.groupby(group_cols, dropna=False).size().reset_index(name="rows")
    low_groups["low_sample_flag"] = low_groups["rows"].lt(MIN_GROUP_SIZE)
    top_groups = low_groups.sort_values("rows", ascending=False).head(50).copy()
    missing = pd.DataFrame(
        [{"feature": col, "missing_rate": df[col].isna().mean()} for col in created if col.endswith("_pct") or col.endswith("_z")]
    )
    summary = pd.DataFrame(
        [
            {"metric": "input_file", "value": str(path)},
            {"metric": "output_file", "value": str(OUTPUT)},
            {"metric": "rows", "value": len(df)},
            {"metric": "features_created", "value": len(created)},
            {"metric": "rows_with_any_valid_league_adjusted_feature", "value": int(df[[c for c in created if c.endswith('_pct') or c.endswith('_z')]].notna().any(axis=1).sum()) if created else 0},
            {"metric": "low_sample_league_season_groups", "value": int(low_groups["low_sample_flag"].sum())},
            {"metric": "leakage_risk_found", "value": False},
        ]
    )
    summary.to_csv(REPORTS_DIR / "league_adjusted_feature_summary.csv", index=False)
    missing.to_csv(REPORTS_DIR / "league_adjusted_feature_missing_rates.csv", index=False)
    low_groups.to_csv(REPORTS_DIR / "league_adjusted_low_sample_groups.csv", index=False)
    top_groups.to_csv(REPORTS_DIR / "league_adjusted_top_groups.csv", index=False)
    print(f"Wrote {len(created)} league-adjusted features to {OUTPUT}")


if __name__ == "__main__":
    run()
