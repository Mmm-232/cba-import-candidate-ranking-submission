"""
Module: export_frontend_recommendations.py
Purpose: Export recommendation and optional biodata outputs required by the Streamlit frontend.
"""
from __future__ import annotations

import argparse
import re
from urllib.parse import quote_plus

import pandas as pd

from ..research import large_scale_rank_utils as lu
from ..research import player_season_rank_utils as ps
from ..utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


OUTPUT = REPORTS_DIR / "frontend_recommendations.csv"
ENRICHED_OUTPUT = REPORTS_DIR / "frontend_recommendations_enriched.csv"
BIODATA_OVERRIDES = PROCESSED_DIR.parent / "manual" / "player_biodata_overrides.csv"
PREDICTION_FILE = REPORTS_DIR / "context_feature_topk_predictions.csv"

SEASON_COLUMNS = ["season", "candidate_season", "test_season", "prediction_season", "target_season"]

EXPORT_FIELDS = [
    "rank",
    "player_name_raw",
    "player_name_key",
    "recommendation_season",
    "league",
    "team",
    "source",
    "source_id",
    "model_setup",
    "score",
    "reason_summary",
    "games",
    "minutes",
    "minutes_per_game",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "points_per_36",
    "usage_proxy",
    "ts_pct",
    "efg_pct",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "data_completeness_score",
    "has_prior_cba_experience_before_t",
    "prior_cba_seasons_before_t",
    "prior_cba_last_seen_gap",
    "has_previous_season_record",
    "points_per_36_trend",
    "usage_proxy_trend",
    "ts_pct_trend",
    "minutes_per_game_trend",
    "height",
    "weight",
    "age",
    "birth_year",
    "birth_date",
    "biodata_source",
    "biodata_notes",
    "position",
    "country",
    "last_affiliation",
    "age_at_recommendation_season",
    "official_player_url",
    "official_stats_url",
    "official_search_url",
    "nba_stats_url",
    "euroleague_profile_or_search_url",
    "league_official_search_url",
    "video_search_query",
    "youtube_search_url",
    "google_video_search_url",
    "youtube_highlight_search_url",
    "cba_video_search_query",
    "basketball_reference_search_url",
    "is_current_recommendation_season",
    "export_scope",
    "input_dataset_used",
]


# 功能：把不同写法的赛季统一为项目使用的格式。
def _normalise_season(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    years = re.findall(r"\d{4}", text)
    if len(years) >= 2:
        return f"{years[0]}-{years[1]}"
    short = re.match(r"^\s*(\d{4})\s*[-/]\s*(\d{2})\s*$", text)
    if short:
        start = int(short.group(1))
        end_suffix = int(short.group(2))
        end = (start // 100) * 100 + end_suffix
        if end < start:
            end += 100
        return f"{start}-{end}"
    if re.match(r"^\d{4}$", text):
        start = int(text)
        return f"{start}-{start + 1}"
    return text.replace(" - ", "-").replace(" / ", "-")


# 功能：从赛季文本中安全提取开始年份。
def _season_start(value: object) -> float:
    normalised = _normalise_season(value)
    match = re.search(r"\d{4}", normalised)
    return float(match.group(0)) if match else float("nan")


# 功能：在输入表中寻找可用的赛季字段。
def _find_season_column(df: pd.DataFrame) -> str:
    for col in SEASON_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return col
    if "season_start_year" in df.columns:
        return "season_start_year"
    raise ValueError("No recommendation season column found.")


# 功能：读取最终排名结果并选择指定赛季的候选人。
def _source_predictions() -> tuple[pd.DataFrame, str, str]:
    if PREDICTION_FILE.exists():
        df = pd.read_csv(PREDICTION_FILE)
        filt = (
            df.get("dataset_pool", "").astype(str).eq("pool_common_cba_source_leagues")
            & df.get("variant", "").astype(str).eq("current_main_baseline")
            & df.get("model", "").astype(str).eq("rule_based")
        )
        picked = df[filt].copy()
        if not picked.empty:
            return picked, str(PREDICTION_FILE), "common CBA source league pool + rule-based baseline"

    path = REPORTS_DIR / "large_scale_final_watchlist.csv"
    if path.exists():
        return pd.read_csv(path), str(path), "validation-selected large-scale watchlist"

    path = REPORTS_DIR / "ltr_topk_predictions.csv"
    if path.exists():
        df = pd.read_csv(path)
        filt = (
            df.get("pool", "").astype(str).eq("common_cba_source_leagues")
            & df.get("model", "").astype(str).eq("rule_based")
        )
        picked = df[filt].copy()
        if not picked.empty:
            return picked, str(path), "common CBA source league pool + rule-based baseline"

    df = pd.read_csv(PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv")
    latest_year = pd.to_numeric(df["season_start_year"], errors="coerce").max()
    df = df[pd.to_numeric(df["season_start_year"], errors="coerce").eq(latest_year)].copy()
    df = df[lu.pool_mask(df, "pool_common_cba_source_leagues")].copy()
    df["score"] = ps.fit_rule_score(df)
    df["rank"] = df["score"].rank(method="first", ascending=False).astype(int)
    return df, str(PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"), "recomputed common CBA source league pool + rule-based baseline"


# 功能：根据球员统计生成简短的推荐理由。
def _reason(row: pd.Series) -> str:
    prior = bool(float(row.get("has_prior_cba_experience_before_t", 0) or 0) > 0)
    missing_perf = pd.isna(row.get("points_per_36")) and pd.isna(row.get("usage_proxy"))
    if prior:
        return "Returning-import candidate with prior CBA experience and strong source-league/pathway signal."
    if missing_perf:
        return "Recommended mainly due to pathway/source signal; performance data should be manually reviewed."
    if float(row.get("usage_proxy", 0) or 0) >= 0.4 or float(row.get("points_per_36", 0) or 0) >= 18:
        return "High-ranked candidate from a common CBA source league with strong usage/scoring profile."
    return "High-ranked candidate from a common CBA source league with interpretable pathway signal."


# 功能：安全读取字段；缺失时生成空列。
def _series_or_blank(df: pd.DataFrame, *cols: str) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return df[col]
    return pd.Series(pd.NA, index=df.index)


# 功能：把球员个人资料合并到推荐名单中。
def _merge_biodata(df: pd.DataFrame) -> pd.DataFrame:
    if not BIODATA_OVERRIDES.exists() or "player_name_key" not in df.columns:
        return df
    bio = pd.read_csv(BIODATA_OVERRIDES)
    if "player_name_key" not in bio.columns:
        return df
    rename = {"notes": "biodata_notes"}
    bio = bio.rename(columns={k: v for k, v in rename.items() if k in bio.columns})
    keep = [
        "player_name_key",
        "height",
        "weight",
        "age",
        "birth_year",
        "birth_date",
        "biodata_source",
        "biodata_notes",
    ]
    keep = [c for c in keep if c in bio.columns]
    existing_bio_cols = [c for c in keep if c != "player_name_key" and c in df.columns]
    if existing_bio_cols:
        df = df.drop(columns=existing_bio_cols)
    return df.merge(bio[keep].drop_duplicates("player_name_key"), on="player_name_key", how="left")


# 功能：整理 Dashboard 使用的最终推荐表。
def build_export(all_seasons: bool = False, top_n: int = 300) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_data_dirs()
    df, source_file, model_setup = _source_predictions()
    rows_before = len(df)
    season_col = _find_season_column(df)
    if season_col == "season_start_year":
        df["recommendation_season"] = pd.to_numeric(df[season_col], errors="coerce").map(
            lambda x: f"{int(x)}-{int(x) + 1}" if pd.notna(x) else pd.NA
        )
    else:
        df["recommendation_season"] = df[season_col].map(_normalise_season)
    df["_season_start"] = df["recommendation_season"].map(_season_start)
    latest_start = df["_season_start"].max()
    latest_season = df.loc[df["_season_start"].eq(latest_start), "recommendation_season"].dropna().astype(str).iloc[0]
    df["is_current_recommendation_season"] = df["_season_start"].eq(latest_start)
    export_scope = "all_seasons" if all_seasons else "latest_season_only"
    if not all_seasons:
        df = df[df["is_current_recommendation_season"]].copy()

    if "rank" not in df.columns:
        df["rank"] = df["score"].rank(method="first", ascending=False).astype(int)
    df = df.sort_values(["_season_start", "rank"], ascending=[False, True]).head(top_n).copy()

    df["source"] = _series_or_blank(df, "source", "source_id", "source_group", "sources_present", "best_source_id")
    df["source_id"] = _series_or_blank(df, "source_id", "best_source_id", "best_underlying_source_id", "source")
    df["team"] = _series_or_blank(df, "team", "best_row_team", "best_underlying_team", "teams_played_that_season")
    df["league"] = _series_or_blank(df, "league", "best_row_league", "best_underlying_league", "leagues_played_that_season")
    df["data_completeness_score"] = _series_or_blank(
        df, "data_completeness_score", "best_row_data_completeness_score", "mean_data_completeness_score"
    )
    df["reason_summary"] = df.apply(_reason, axis=1)
    df["video_search_query"] = [
        f"{row.player_name_raw} {row.league} basketball highlights" if pd.notna(row.league) else f"{row.player_name_raw} basketball highlights"
        for row in df.itertuples(index=False)
    ]
    df["youtube_search_url"] = "https://www.youtube.com/results?search_query=" + df["video_search_query"].map(quote_plus)
    df["google_video_search_url"] = "https://www.google.com/search?tbm=vid&q=" + df["video_search_query"].map(quote_plus)
    df["youtube_highlight_search_url"] = df["youtube_search_url"]
    df["basketball_reference_search_url"] = [
        "https://www.google.com/search?q=" + quote_plus(f'site:basketball-reference.com "{row.player_name_raw}" basketball')
        for row in df.itertuples(index=False)
    ]
    df["cba_video_search_query"] = [
        f"{row.player_name_raw} CBA basketball highlights"
        if float(getattr(row, "has_prior_cba_experience_before_t", 0) or 0) > 0
        else pd.NA
        for row in df.itertuples(index=False)
    ]
    df["model_setup"] = model_setup
    df["input_dataset_used"] = source_file
    df["export_scope"] = export_scope

    df = _merge_biodata(df)
    for col in EXPORT_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA

    report = {
        "latest_recommendation_season": latest_season,
        "rows_before_filtering": rows_before,
        "rows_after_filtering": len(df),
        "positive_rows": int(pd.to_numeric(df.get("signed_cba_next_season", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if "signed_cba_next_season" in df.columns
        else "not_available",
        "top_n_exported": min(top_n, len(df)),
        "export_scope": export_scope,
        "season_column": season_col,
        "biodata_override_used": BIODATA_OVERRIDES.exists(),
    }
    return df[EXPORT_FIELDS], report


# 功能：保留已有推荐文件中已经补充的资料字段。
def _merge_existing_enriched(out: pd.DataFrame) -> pd.DataFrame:
    if not ENRICHED_OUTPUT.exists():
        print(f"No enriched recommendation file found at {ENRICHED_OUTPUT}; exported base recommendations only.")
        return out
    enriched = pd.read_csv(ENRICHED_OUTPUT)
    enrichment_cols = [
        "player_name_key",
        "height",
        "weight",
        "birthdate",
        "age",
        "birth_year",
        "birth_date",
        "position",
        "country",
        "last_affiliation",
        "age_at_recommendation_season",
        "official_player_url",
        "official_stats_url",
        "official_search_url",
        "nba_stats_url",
        "euroleague_profile_or_search_url",
        "league_official_search_url",
        "basketball_reference_search_url",
        "biodata_source",
        "biodata_match_confidence",
        "biodata_match_method",
    ]
    enrichment_cols = [c for c in enrichment_cols if c in enriched.columns]
    if "player_name_key" not in enrichment_cols:
        return out
    keep = enriched[enrichment_cols].drop_duplicates("player_name_key", keep="last")
    merged = out.drop(columns=[c for c in enrichment_cols if c != "player_name_key" and c in out.columns], errors="ignore")
    return merged.merge(keep, on="player_name_key", how="left")


# 功能：导出 Dashboard 使用的最终推荐文件。
def run(all_seasons: bool = False, top_n: int = 300, with_biodata: bool = False) -> None:
    out, report = build_export(all_seasons=all_seasons, top_n=top_n)
    if with_biodata:
        out = _merge_existing_enriched(out)
    out.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(out)} frontend recommendations to {OUTPUT}")
    print(f"Latest recommendation season selected: {report['latest_recommendation_season']}")
    print(f"Rows before filtering: {report['rows_before_filtering']}")
    print(f"Rows after filtering: {report['rows_after_filtering']}")
    print(f"Positive rows: {report['positive_rows']}")
    print(f"Top N exported: {report['top_n_exported']}")
    print(f"Export scope: {report['export_scope']}")


# 功能：导出 Dashboard 使用的最终推荐文件。
def main() -> None:
    parser = argparse.ArgumentParser(description="Export Streamlit dashboard recommendation candidates.")
    parser.add_argument("--all-seasons", action="store_true", help="Export all validation seasons instead of latest recommendation season only.")
    parser.add_argument("--top-n", type=int, default=300, help="Maximum rows to export.")
    parser.add_argument("--with-biodata", action="store_true", help="Merge display-only enriched biodata/link fields if the enriched file exists.")
    args = parser.parse_args()
    run(all_seasons=args.all_seasons, top_n=args.top_n, with_biodata=args.with_biodata)


if __name__ == "__main__":
    main()
