from __future__ import annotations

import pandas as pd

try:
    from ..sources.gleague_nba_api_source import GLeagueNbaApiSource
    from .utils import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_data_dirs, season_start_year
except ImportError:
    from sources.gleague_nba_api_source import GLeagueNbaApiSource
    from utils import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_data_dirs, season_start_year


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
    "usage_rate",
    "stat_scale_source",
    "advanced_stat_scale_source",
    "source_confidence",
    "verification_status",
    "signed_cba_next_season",
]


# 函数：_label_path
def _label_path() -> tuple[pd.DataFrame, str]:
    path = PROCESSED_DIR / "cba_imports_extended_verified.csv"
    if not path.exists():
        path = PROCESSED_DIR / "cba_imports_extended.csv"
    return pd.read_csv(path), path.name


# 函数：_label
def _label(pool: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    labels, label_file = _label_path()
    pairs = set(zip(labels["player_name_key"].astype(str), labels["cba_season"].astype(str)))
    out = pool.copy()
    out["signed_cba_next_season"] = [
        int((str(row.player_name_key), str(row.next_season)) in pairs) for row in out.itertuples(index=False)
    ]
    return out, label_file


# 函数：_current_positive_pairs
def _current_positive_pairs() -> set[tuple[str, str]]:
    candidates = [
        PROCESSED_DIR / "labelled_candidate_dataset_multisource_verified.csv",
        PROCESSED_DIR / "labelled_candidate_dataset_multisource.csv",
        PROCESSED_DIR / "labelled_candidate_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"player_name_key", "next_season", "signed_cba_next_season"}.issubset(df.columns):
                pos = df[df["signed_cba_next_season"].eq(1)]
                return set(zip(pos["player_name_key"].astype(str), pos["next_season"].astype(str)))
    return set()


# 函数：_year
def _year(value: object) -> int | None:
    try:
        return season_start_year(str(value))
    except ValueError:
        return None


# 函数：_write_unavailable_report
def _write_unavailable_report(raw: pd.DataFrame) -> None:
    rows = [
        {"metric": "status", "value": "unavailable_or_empty"},
        {"metric": "raw_rows", "value": len(raw)},
        {"metric": "mapped_rows", "value": 0},
        {"metric": "positives", "value": 0},
        {"metric": "note", "value": "No valid G League rows were returned from nba_api LeagueID=20; local CSV fallback remains required."},
    ]
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "gleague_nba_api_source_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "gleague_nba_api_label_summary.csv", index=False)


# 函数：_write_reports
def _write_reports(pool: pd.DataFrame, label_file: str) -> None:
    current_pairs = _current_positive_pairs()
    positives = pool[pool["signed_cba_next_season"].eq(1)].copy()
    positive_pairs = set(zip(positives["player_name_key"].astype(str), positives["next_season"].astype(str)))
    new_positive_pairs = positive_pairs - current_pairs
    recent = positives[positives["next_season"].map(_year).fillna(0).ge(2018)]

    summary_rows = [
        {"metric": "status", "value": "success"},
        {"metric": "source_id", "value": "gleague_nba_api"},
        {"metric": "label_file", "value": label_file},
        {"metric": "total_gleague_rows", "value": len(pool)},
        {"metric": "unique_gleague_players", "value": pool["player_name_key"].nunique()},
        {"metric": "seasons_covered", "value": "; ".join(sorted(pool["season"].dropna().astype(str).unique()))},
        {"metric": "positives_found", "value": int(pool["signed_cba_next_season"].sum())},
        {"metric": "new_positive_player_next_season_pairs", "value": len(new_positive_pairs)},
        {"metric": "recent_2018_plus_positive_rows", "value": len(recent)},
    ]
    pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "gleague_nba_api_source_summary.csv", index=False)

    by_season = positives.groupby("next_season", dropna=False).agg(
        positives=("player_name_key", "size"),
        unique_players=("player_name_key", "nunique"),
        players=("player_name_raw", lambda s: "; ".join(sorted(s.dropna().astype(str).unique()))),
    ).reset_index()
    by_player = positives.groupby(["player_name_key", "player_name_raw"], dropna=False).agg(
        positive_rows=("season", "size"),
        seasons=("season", lambda s: "; ".join(sorted(s.dropna().astype(str).unique()))),
        next_seasons=("next_season", lambda s: "; ".join(sorted(s.dropna().astype(str).unique()))),
    ).reset_index()
    label_rows = summary_rows + [{"metric": "positives_by_season", "value": by_season.to_json(orient="records")}, {"metric": "positives_by_player", "value": by_player.to_json(orient="records")}]
    pd.DataFrame(label_rows).to_csv(REPORTS_DIR / "gleague_nba_api_label_summary.csv", index=False)


# 函数：_write_sanity_check
def _write_sanity_check(pool: pd.DataFrame) -> None:
    checks = pool.copy()
    for col in ["minutes_per_game", "points_per_36", "usage_proxy", "usage_rate", "ts_pct", "efg_pct"]:
        checks[col] = pd.to_numeric(checks.get(col), errors="coerce")
    checks["flag_minutes_per_game_out_of_range"] = checks["minutes_per_game"].lt(1) | checks["minutes_per_game"].gt(45)
    active = checks["minutes_per_game"].ge(10)
    checks["flag_points_per_36_low_active"] = active & checks["points_per_36"].lt(1)
    checks["flag_usage_proxy_low_active"] = active & checks["usage_proxy"].lt(0.05)
    checks["flag_ts_pct_out_of_range"] = checks["ts_pct"].lt(0) | checks["ts_pct"].gt(1.2)
    checks["flag_efg_pct_out_of_range"] = checks["efg_pct"].lt(0) | checks["efg_pct"].gt(1.2)
    flag_cols = [c for c in checks.columns if c.startswith("flag_")]
    suspicious = checks[checks[flag_cols].any(axis=1)].copy()
    suspicious[
        [
            "player_name_raw",
            "season",
            "team",
            "minutes_per_game",
            "points_per_36",
            "usage_proxy",
            "usage_rate",
            "ts_pct",
            "efg_pct",
            *flag_cols,
        ]
    ].to_csv(REPORTS_DIR / "gleague_stat_sanity_check.csv", index=False)

    dist = checks[["minutes_per_game", "points_per_36", "usage_proxy", "usage_rate", "ts_pct", "efg_pct"]].describe().T
    lines = [
        "# G League Stat Sanity Check",
        "",
        f"- Rows checked: {len(checks)}",
        f"- Suspicious rows: {len(suspicious)}",
        f"- Active rows with points_per_36 < 1: {int(checks['flag_points_per_36_low_active'].sum())}",
        f"- Active rows with usage_proxy < 0.05: {int(checks['flag_usage_proxy_low_active'].sum())}",
        "",
        "```csv",
        dist.to_csv().strip(),
        "```",
    ]
    (REPORTS_DIR / "gleague_stat_sanity_check.md").write_text("\n".join(lines), encoding="utf-8")


# 函数：run
def run() -> pd.DataFrame:
    ensure_data_dirs()
    source = GLeagueNbaApiSource()
    raw = source.collect_raw()
    raw.to_csv(RAW_DIR / "gleague_nba_api_candidate_pool_raw.csv", index=False)
    pool = source.collect_mapped()
    if pool.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(PROCESSED_DIR / "gleague_nba_api_candidate_pool_mapped.csv", index=False)
        _write_unavailable_report(raw)
        print("G League NBA API candidate pool unavailable or empty.")
        return pool
    pool, label_file = _label(pool)
    for col in OUTPUT_COLUMNS:
        if col not in pool.columns:
            pool[col] = pd.NA
    pool[OUTPUT_COLUMNS].to_csv(PROCESSED_DIR / "gleague_nba_api_candidate_pool_mapped.csv", index=False)
    _write_reports(pool, label_file)
    _write_sanity_check(pool)
    print(f"Mapped G League NBA API candidate pool: {len(pool)} rows, {int(pool['signed_cba_next_season'].sum())} positives")
    return pool


# 函数：main
def main() -> None:
    run()


if __name__ == "__main__":
    main()
