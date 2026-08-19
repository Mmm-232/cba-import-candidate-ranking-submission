from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from ..research import large_scale_rank_utils as lu
from ..research import player_season_rank_utils as ps
from ..utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


DEFAULT_INPUT = PROCESSED_DIR / "new_candidates_clean.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "new_candidate_recommendations.csv"


def _percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index)
    return values.rank(pct=True).fillna(0.0)


def _common_pathway_flag(df: pd.DataFrame) -> pd.Series:
    return df.apply(lu.is_common_pathway, axis=1).astype(int)


def _prior_cba_flag(df: pd.DataFrame) -> pd.Series:
    for col in ["has_prior_cba_experience_before_t", "has_prior_cba_experience", "prior_cba_experience"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).astype(int)
    return pd.Series(0, index=df.index)


def _reason(row: pd.Series) -> str:
    parts = []
    if row.get("score_component_pathway", 0) > 0:
        parts.append("common CBA source league/pathway")
    if pd.to_numeric(pd.Series([row.get("points_per_36")]), errors="coerce").fillna(0).iloc[0] >= 18:
        parts.append("strong scoring per 36")
    if pd.to_numeric(pd.Series([row.get("usage_proxy")]), errors="coerce").fillna(0).iloc[0] >= 0.45:
        parts.append("high usage proxy")
    if pd.to_numeric(pd.Series([row.get("ts_pct")]), errors="coerce").fillna(0).iloc[0] >= 0.58:
        parts.append("efficient scoring profile")
    if row.get("score_component_context", 0) > 0:
        parts.append("prior CBA context supplied")
    if not parts:
        parts.append("ranked by available transparent rule-based scouting features")
    warning_value = row.get("data_quality_warning", "")
    warning = "" if pd.isna(warning_value) else str(warning_value).strip()
    if warning and warning.lower() not in {"nan", "none"}:
        parts.append(f"data warning: {warning}")
    return "; ".join(parts)


def rank_candidates(input_path: str | Path = DEFAULT_INPUT, output_path: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    ensure_data_dirs()
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)
    if df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    df["has_common_pathway_league"] = _common_pathway_flag(df)
    df["score_component_pathway"] = df["has_common_pathway_league"].astype(float)
    perf_parts = [
        _percentile(df.get("points_per_36", pd.Series(index=df.index, dtype=float))),
        _percentile(df.get("usage_proxy", pd.Series(index=df.index, dtype=float))),
        _percentile(df.get("ts_pct", pd.Series(index=df.index, dtype=float))),
        _percentile(df.get("minutes_per_game", pd.Series(index=df.index, dtype=float))),
    ]
    df["score_component_performance"] = pd.concat(perf_parts, axis=1).mean(axis=1).fillna(0.0)
    df["score_component_context"] = 0.5 * _prior_cba_flag(df)
    df["score_component_completeness"] = pd.to_numeric(df.get("data_completeness_score"), errors="coerce").fillna(0.0)

    # Apply the dissertation's fixed three-term rule; other components remain available for review only.
    if "cba_import_fit_score" in df.columns:
        fit_signal = pd.to_numeric(df["cba_import_fit_score"], errors="coerce").fillna(0.0)
    else:
        fit_signal = df["score_component_pathway"]

    base_rule = ps.fit_rule_score(
        df.assign(
            cba_import_fit_score=fit_signal,
            max_usage_proxy=df.get("usage_proxy"),
            max_points_per_36=df.get("points_per_36"),
        )
    )
    df["recommendation_score"] = pd.to_numeric(base_rule, errors="coerce").fillna(0)
    df["new_rank"] = df["recommendation_score"].rank(method="first", ascending=False).astype(int)
    df["reason_summary"] = df.apply(_reason, axis=1)
    df["video_search_query"] = [
        f"{row.player_name_raw} {row.league} basketball highlights" if pd.notna(row.league) else f"{row.player_name_raw} basketball highlights"
        for row in df.itertuples(index=False)
    ]
    df["youtube_search_url"] = "https://www.youtube.com/results?search_query=" + df["video_search_query"].map(quote_plus)
    df["google_video_search_url"] = "https://www.google.com/search?tbm=vid&q=" + df["video_search_query"].map(quote_plus)

    output_cols = [
        "new_rank",
        "player_name_raw",
        "player_name_key",
        "season",
        "league",
        "team",
        "source",
        "recommendation_score",
        "score_component_pathway",
        "score_component_performance",
        "score_component_context",
        "score_component_completeness",
        "reason_summary",
        "games",
        "minutes_per_game",
        "points_per_36",
        "usage_proxy",
        "ts_pct",
        "fg_pct",
        "three_pct",
        "ft_pct",
        "height",
        "weight",
        "age",
        "official_player_url",
        "official_stats_url",
        "youtube_search_url",
        "google_video_search_url",
        "video_search_query",
        "data_quality_warning",
    ]
    for col in output_cols:
        if col not in df.columns:
            df[col] = pd.NA
    ranked = df.sort_values("new_rank")[output_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    print(f"Wrote new candidate recommendations to {output_path}")
    print(f"Rows ranked: {len(ranked)}")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank new uploaded CBA import candidate player-season rows.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Cleaned new candidates CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Recommendation output CSV.")
    args = parser.parse_args()
    rank_candidates(args.input, args.output)


if __name__ == "__main__":
    main()
