from __future__ import annotations

from io import StringIO

import pandas as pd

from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


INPUTS = [
    PROCESSED_DIR / "labelled_candidate_dataset_multisource_domestic.csv",
    PROCESSED_DIR / "labelled_candidate_dataset_multisource_gleague_with_features.csv",
    PROCESSED_DIR / "labelled_candidate_dataset_multisource_gleague.csv",
    PROCESSED_DIR / "labelled_candidate_dataset_with_cba_fit.csv",
    PROCESSED_DIR / "labelled_candidate_dataset_multisource_verified.csv",
]


# 功能：读取训练或评估所需的最终候选数据。
def _load() -> pd.DataFrame:
    selected_path = None
    for path in INPUTS:
        if path.exists():
            df = pd.read_csv(path)
            selected_path = path
            break
    else:
        raise FileNotFoundError("Missing row-level candidate dataset.")
    df.attrs["selected_input_path"] = str(selected_path)
    if "season_start_year" not in df.columns:
        df["season_start_year"] = pd.to_numeric(df["season"].astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
    for col in [
        "games",
        "minutes",
        "minutes_per_game",
        "points_per_36",
        "usage_proxy",
        "ts_pct",
        "cba_import_fit_score",
        "data_completeness_score",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "role_cluster_label" not in df.columns:
        df["role_cluster_label"] = "unknown_role"
    return df.reset_index(drop=False).rename(columns={"index": "underlying_row_id"})


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _uniq(values: pd.Series) -> str:
    vals = [str(v) for v in values.dropna().astype(str) if str(v).strip() and str(v).lower() != "nan"]
    return "; ".join(sorted(set(vals)))


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _first(values: pd.Series):
    vals = values.dropna()
    return vals.iloc[0] if not vals.empty else pd.NA


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _weighted_avg(group: pd.DataFrame, col: str) -> float:
    values = pd.to_numeric(group[col], errors="coerce")
    weights = pd.to_numeric(group.get("minutes", pd.Series(index=group.index, dtype=float)), errors="coerce")
    if weights.notna().sum() == 0 or weights.fillna(0).sum() <= 0:
        weights = pd.to_numeric(group.get("games", pd.Series(index=group.index, dtype=float)), errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if valid.any():
        return float((values[valid] * weights[valid]).sum() / weights[valid].sum())
    return float(values.mean()) if values.notna().any() else float("nan")


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _has_token(values: pd.Series, token: str) -> bool:
    return values.fillna("").astype(str).str.lower().str.contains(token.lower(), regex=False).any()


# 功能：处理当前步骤所需的数据，并返回整理后的结果。
def _aggregate_group(group: pd.DataFrame) -> dict:
    completeness = pd.to_numeric(group["data_completeness_score"], errors="coerce").fillna(-1)
    fit = pd.to_numeric(group["cba_import_fit_score"], errors="coerce").fillna(-1)
    best_idx = (completeness + fit / 10).idxmax()
    best = group.loc[best_idx]
    league_median_usage = pd.to_numeric(group["usage_proxy"], errors="coerce").median()
    row = {
        "player_name_raw": group["player_name_raw"].mode().iloc[0] if group["player_name_raw"].notna().any() else _first(group["player_name_raw"]),
        "player_name_key": _first(group["player_name_key"]),
        "season": _first(group["season"]),
        "next_season": _first(group["next_season"]),
        "season_start_year": _first(group["season_start_year"]),
        "signed_cba_next_season": int(group["signed_cba_next_season"].max()),
        "leagues_played_that_season": _uniq(group["league"]),
        "teams_played_that_season": _uniq(group["team"]),
        "sources_present": _uniq(group["source_id"]),
        "best_source_id": best.get("source_id", pd.NA),
        "highest_completeness_league": best.get("league", pd.NA),
        "number_of_leagues_that_season": group["league"].dropna().astype(str).nunique(),
        "multi_league_season_flag": int(group["league"].dropna().astype(str).nunique() > 1),
        "row_count_for_player_season": len(group),
        "best_row_data_completeness_score": pd.to_numeric(best.get("data_completeness_score"), errors="coerce"),
        "mean_data_completeness_score": pd.to_numeric(group["data_completeness_score"], errors="coerce").mean(),
        "has_euroleague_row": int(_has_token(group["league"], "Euroleague") or _has_token(group["league"], "EuroLeague")),
        "has_eurocup_row": int(_has_token(group["league"], "Eurocup") or _has_token(group["league"], "EuroCup")),
        "has_gleague_row": int(_has_token(group["league"], "G League") or _has_token(group["league"], "G-League")),
        "gleague_team": _uniq(group.loc[group["league"].fillna("").astype(str).str.contains("G League", case=False, na=False), "team"]) if "team" in group else pd.NA,
        "gleague_source_id": _uniq(group.loc[group["league"].fillna("").astype(str).str.contains("G League", case=False, na=False), "source_id"]) if "source_id" in group else pd.NA,
        "gleague_data_completeness_score": pd.to_numeric(
            group.loc[group["league"].fillna("").astype(str).str.contains("G League", case=False, na=False), "data_completeness_score"],
            errors="coerce",
        ).max(),
        "has_common_pathway_league": int(group.get("common_cba_source_league_flag", pd.Series(0, index=group.index)).fillna(0).astype(float).gt(0).any()),
        "has_high_usage_row": int(pd.to_numeric(group["usage_proxy"], errors="coerce").gt(league_median_usage).any()),
        "has_top_fit_decile_row": int(pd.to_numeric(group["cba_import_fit_score"], errors="coerce").rank(pct=True).ge(0.9).any()),
        "best_underlying_row_id": best.get("underlying_row_id", pd.NA),
        "best_underlying_source_id": best.get("source_id", pd.NA),
        "best_underlying_league": best.get("league", pd.NA),
        "best_underlying_team": best.get("team", pd.NA),
        "best_row_cba_import_fit_score": best.get("cba_import_fit_score", pd.NA),
        "best_row_role_cluster_label": best.get("role_cluster_label", "unknown_role"),
        "best_row_league": best.get("league", pd.NA),
        "best_row_team": best.get("team", pd.NA),
    }
    for col in ["minutes_per_game", "points_per_36", "usage_proxy", "ts_pct", "cba_import_fit_score"]:
        row[f"max_{col}"] = pd.to_numeric(group[col], errors="coerce").max()
        row[f"weighted_avg_{col}"] = _weighted_avg(group, col)
    # Aliases make existing modelling helpers easier to reuse.
    row["minutes_per_game"] = row["weighted_avg_minutes_per_game"]
    row["points_per_36"] = row["weighted_avg_points_per_36"]
    row["usage_proxy"] = row["weighted_avg_usage_proxy"]
    row["ts_pct"] = row["weighted_avg_ts_pct"]
    row["cba_import_fit_score"] = row["weighted_avg_cba_import_fit_score"]
    row["league"] = row["best_row_league"]
    row["source_id"] = row["best_underlying_source_id"]
    row["role_cluster_label"] = row["best_row_role_cluster_label"]
    return row


# 功能：执行多来源球员赛季聚合并保存最终候选数据。
def run() -> None:
    ensure_data_dirs()
    df = _load()
    selected_input = str(df.attrs.get("selected_input_path", ""))
    is_domestic_run = "multisource_domestic" in selected_input
    is_gleague_run = (PROCESSED_DIR / "labelled_candidate_dataset_multisource_gleague.csv").exists() and "multisource_gleague" in selected_input
    grouped = [_aggregate_group(group) for _, group in df.groupby(["player_name_key", "season"], dropna=False)]
    out = pd.DataFrame(grouped)
    if is_domestic_run:
        output_path = PROCESSED_DIR / "labelled_player_season_dataset_domestic.csv"
    else:
        output_path = PROCESSED_DIR / ("labelled_player_season_dataset_gleague.csv" if is_gleague_run else "labelled_player_season_dataset.csv")
    out.to_csv(output_path, index=False)

    original_rows = len(df)
    aggregated_rows = len(out)
    original_pos = int(df["signed_cba_next_season"].sum())
    aggregated_pos = int(out["signed_cba_next_season"].sum())
    multi = out.sort_values("row_count_for_player_season", ascending=False).head(30)
    summary = pd.DataFrame(
        [
            {"metric": "original_row_count", "value": original_rows},
            {"metric": "aggregated_player_season_row_count", "value": aggregated_rows},
            {"metric": "original_positive_rows", "value": original_pos},
            {"metric": "aggregated_positive_player_seasons", "value": aggregated_pos},
            {"metric": "duplicate_reduction_percentage", "value": round((1 - aggregated_rows / original_rows) * 100, 3)},
            {"metric": "positives_retained", "value": aggregated_pos},
            {"metric": "multi_row_player_seasons", "value": int(out["row_count_for_player_season"].gt(1).sum())},
        ]
    )
    summary_name = "player_season_aggregation_domestic_summary.csv" if is_domestic_run else ("player_season_aggregation_gleague_summary.csv" if is_gleague_run else "player_season_aggregation_summary.csv")
    reduction_name = "player_season_domestic_duplicate_reduction.csv" if is_domestic_run else ("player_season_gleague_duplicate_reduction.csv" if is_gleague_run else "player_season_duplicate_reduction.csv")
    summary.to_csv(REPORTS_DIR / summary_name, index=False)
    multi[
        [
            "player_name_raw",
            "season",
            "row_count_for_player_season",
            "leagues_played_that_season",
            "signed_cba_next_season",
            "best_row_league",
            "best_row_cba_import_fit_score",
        ]
    ].to_csv(REPORTS_DIR / reduction_name, index=False)

    buffer = StringIO()
    summary.to_csv(buffer, index=False)
    md_name = "player_season_aggregation_domestic_summary.md" if is_domestic_run else ("player_season_aggregation_gleague_summary.md" if is_gleague_run else "player_season_aggregation_summary.md")
    (REPORTS_DIR / md_name).write_text(
        "# Player-Season Aggregation Summary\n\n```csv\n" + buffer.getvalue().strip() + "\n```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
