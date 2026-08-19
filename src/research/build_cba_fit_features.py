from __future__ import annotations

import numpy as np
import pandas as pd

from .literature_rank_utils import ensure_model_columns, split_years, zscore
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


def _source_score(df: pd.DataFrame, col: str) -> pd.Series:
    return zscore(df[col]).rank(pct=True).fillna(0.5)


def _similarity_to_imports(train_pos: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.Series:
    if train_pos.empty:
        return pd.Series(0.0, index=test.index)
    center = train_pos[cols].apply(pd.to_numeric, errors="coerce").median()
    spread = train_pos[cols].apply(pd.to_numeric, errors="coerce").std().replace(0, np.nan).fillna(1)
    dist = (((test[cols].apply(pd.to_numeric, errors="coerce") - center) / spread) ** 2).sum(axis=1).pow(0.5)
    return (1 / (1 + dist)).fillna(0)


def _add_train_only_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["role_positive_rate_train_only", "league_positive_rate_train_only", "source_positive_rate_train_only", "role_similarity_to_historical_cba_imports", "league_pathway_score_train_only"]:
        df[col] = 0.0
    sim_cols = ["points_per_36", "assists_per_36", "rebounds_per_36", "usage_proxy", "ts_pct", "minutes_per_game"]
    for year in sorted(df["season_start_year"].dropna().unique()):
        idx = df["season_start_year"].eq(year)
        train = df[df["season_start_year"] < year]
        if train.empty:
            continue
        global_rate = train["signed_cba_next_season"].mean()
        for group_col, out_col in [
            ("role_cluster_label", "role_positive_rate_train_only"),
            ("league", "league_positive_rate_train_only"),
            ("source_id", "source_positive_rate_train_only"),
        ]:
            rates = train.groupby(group_col)["signed_cba_next_season"].mean()
            df.loc[idx, out_col] = df.loc[idx, group_col].map(rates).fillna(global_rate)
        df.loc[idx, "league_pathway_score_train_only"] = df.loc[idx, "league_positive_rate_train_only"].rank(pct=True).fillna(0.5)
        df.loc[idx, "role_similarity_to_historical_cba_imports"] = _similarity_to_imports(train[train["signed_cba_next_season"] == 1], df.loc[idx], sim_cols)
    return df


def run() -> None:
    ensure_data_dirs()
    path = PROCESSED_DIR / "labelled_candidate_dataset_multisource_gleague_with_roles.csv"
    is_gleague_run = path.exists()
    if not is_gleague_run:
        path = PROCESSED_DIR / "labelled_candidate_dataset_with_roles.csv"
    if not path.exists():
        raise FileNotFoundError("Run python -m src.build_role_style_features first.")
    df = ensure_model_columns(pd.read_csv(path))
    if "role_cluster_label" not in df.columns:
        df["role_cluster_label"] = "unknown_role"
    df = _add_train_only_rates(df)

    df["high_usage_creation_score"] = (
        0.35 * _source_score(df, "usage_proxy")
        + 0.30 * _source_score(df, "points_per_36")
        + 0.20 * _source_score(df, "assists_per_36")
        + 0.15 * _source_score(df, "free_throw_attempt_rate")
    )
    df["scoring_load_score"] = (
        0.40 * _source_score(df, "points_per_36")
        + 0.35 * _source_score(df, "shot_attempts_per_36")
        + 0.25 * _source_score(df, "usage_proxy")
    )
    df["efficiency_stability_score"] = (
        0.45 * _source_score(df, "ts_pct")
        + 0.20 * _source_score(df, "games")
        + 0.20 * _source_score(df, "minutes_per_game")
        + 0.15 * _source_score(df, "data_completeness_score")
    )
    df["role_stability_score"] = 0.55 * _source_score(df, "games") + 0.45 * _source_score(df, "minutes_per_game")
    df["cba_import_fit_score"] = (
        0.24 * df["role_similarity_to_historical_cba_imports"]
        + 0.22 * df["league_pathway_score_train_only"]
        + 0.22 * df["high_usage_creation_score"]
        + 0.18 * df["scoring_load_score"]
        + 0.14 * df["efficiency_stability_score"]
    ).fillna(0)

    if is_gleague_run:
        df.to_csv(PROCESSED_DIR / "labelled_candidate_dataset_multisource_gleague_with_features.csv", index=False)
    else:
        df.to_csv(PROCESSED_DIR / "labelled_candidate_dataset_with_cba_fit.csv", index=False)
    summary_cols = [
        "role_positive_rate_train_only",
        "league_positive_rate_train_only",
        "source_positive_rate_train_only",
        "role_similarity_to_historical_cba_imports",
        "league_pathway_score_train_only",
        "high_usage_creation_score",
        "scoring_load_score",
        "efficiency_stability_score",
        "role_stability_score",
        "cba_import_fit_score",
    ]
    summary_name = "gleague_cba_fit_feature_summary.csv" if is_gleague_run else "cba_fit_feature_summary.csv"
    decile_name = "gleague_cba_fit_positive_rate_by_decile.csv" if is_gleague_run else "cba_fit_positive_rate_by_decile.csv"
    df[summary_cols].describe().T.to_csv(REPORTS_DIR / summary_name)
    dec = df.copy()
    dec["cba_fit_decile"] = pd.qcut(dec["cba_import_fit_score"].rank(method="first"), 10, labels=False, duplicates="drop") + 1
    decile = dec.groupby("cba_fit_decile", as_index=False).agg(rows=("player_name_key", "size"), positives=("signed_cba_next_season", "sum"), avg_fit_score=("cba_import_fit_score", "mean"))
    decile["positive_rate"] = decile["positives"] / decile["rows"]
    decile.to_csv(REPORTS_DIR / decile_name, index=False)
    if is_gleague_run:
        from . import build_player_season_dataset

        build_player_season_dataset.run()


if __name__ == "__main__":
    run()
