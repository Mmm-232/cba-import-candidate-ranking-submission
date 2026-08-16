from __future__ import annotations

import pandas as pd

from .source_diagnostics_utils import add_source_group
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


# 函数：_z
def _z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


# 函数：_pct
def _pct(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True).fillna(0.5)


# 函数：_within
def _within(df: pd.DataFrame, group_cols: list[str], col: str, suffix: str) -> None:
    df[f"{suffix}_{col}_z"] = df.groupby(group_cols, dropna=False)[col].transform(_z)
    df[f"{suffix}_{col}_pct"] = df.groupby(group_cols, dropna=False)[col].transform(_pct)


# 函数：run
def run() -> None:
    ensure_data_dirs()
    input_path = PROCESSED_DIR / "labelled_player_season_dataset_domestic.csv"
    is_domestic_run = input_path.exists()
    if not is_domestic_run:
        input_path = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
    df = add_source_group(pd.read_csv(input_path))
    for col in ["minutes_per_game", "points_per_36", "usage_proxy", "ts_pct", "weighted_avg_ts_pct"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["assists_per_36"] = pd.to_numeric(df.get("assists_per_36", df.get("weighted_avg_assists_per_36")), errors="coerce")
    df["rebounds_per_36"] = pd.to_numeric(df.get("rebounds_per_36", df.get("weighted_avg_rebounds_per_36")), errors="coerce")
    for col in ["minutes_per_game", "points_per_36", "usage_proxy", "ts_pct", "assists_per_36", "rebounds_per_36"]:
        if col not in df:
            df[col] = pd.NA

    mapping = {
        "minutes": "minutes_per_game",
        "points_per_36": "points_per_36",
        "usage": "usage_proxy",
        "ts": "ts_pct",
        "assists_per_36": "assists_per_36",
        "rebounds_per_36": "rebounds_per_36",
    }
    for name, col in mapping.items():
        df[f"league_season_{name}_z"] = df.groupby(["league", "season"], dropna=False)[col].transform(_z)
        df[f"league_season_{name}_pct"] = df.groupby(["league", "season"], dropna=False)[col].transform(_pct)
    for name, col in {"minutes": "minutes_per_game", "points_per_36": "points_per_36", "usage": "usage_proxy", "ts": "ts_pct"}.items():
        df[f"source_season_{name}_pct"] = df.groupby(["source_group", "season"], dropna=False)[col].transform(_pct)
    df["league_season_ts_pct_rank"] = df["league_season_ts_pct"]
    df["source_season_ts_pct_rank"] = df["source_season_ts_pct"]
    df["league_season_creation_pct"] = df[["league_season_usage_pct", "league_season_assists_per_36_pct"]].mean(axis=1)
    df["league_season_scoring_load_pct"] = df[["league_season_points_per_36_pct", "league_season_usage_pct"]].mean(axis=1)
    df["role_cluster_usage_pct"] = df.groupby(["role_cluster_label", "season"], dropna=False)["usage_proxy"].transform(_pct)
    df["role_cluster_scoring_pct"] = df.groupby(["role_cluster_label", "season"], dropna=False)["points_per_36"].transform(_pct)
    df["role_cluster_efficiency_pct"] = df.groupby(["role_cluster_label", "season"], dropna=False)["ts_pct"].transform(_pct)
    df["cross_league_scoring_score"] = df[["league_season_points_per_36_pct", "source_season_points_per_36_pct", "role_cluster_scoring_pct"]].mean(axis=1)
    df["cross_league_creation_score"] = df[["league_season_creation_pct", "source_season_usage_pct", "role_cluster_usage_pct"]].mean(axis=1)
    df["cross_league_efficiency_score"] = df[["league_season_ts_pct_rank", "source_season_ts_pct_rank", "role_cluster_efficiency_pct"]].mean(axis=1)
    df["cross_league_role_score"] = df[["cross_league_scoring_score", "cross_league_creation_score", "cross_league_efficiency_score"]].mean(axis=1)
    df["cross_league_cba_fit_score"] = (
        0.40 * pd.to_numeric(df.get("cba_import_fit_score"), errors="coerce").fillna(0)
        + 0.25 * df["cross_league_scoring_score"]
        + 0.20 * df["cross_league_creation_score"]
        + 0.15 * df["cross_league_efficiency_score"]
    )
    output_path = PROCESSED_DIR / ("labelled_player_season_dataset_domestic_normalised.csv" if is_domestic_run else "labelled_player_season_dataset_gleague_normalised.csv")
    df.to_csv(output_path, index=False)
    summary_cols = [c for c in df.columns if c.startswith("league_season_") or c.startswith("source_season_") or c.startswith("cross_league_") or c.startswith("role_cluster_")]
    summary_name = "cross_league_feature_domestic_summary.csv" if is_domestic_run else "cross_league_feature_summary.csv"
    df[summary_cols].describe().T.to_csv(REPORTS_DIR / summary_name)
    print("Wrote cross-league normalised features")


if __name__ == "__main__":
    run()
