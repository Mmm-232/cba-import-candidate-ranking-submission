from __future__ import annotations

import pandas as pd

from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs, season_start_year


MAIN_INPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
OUTPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague_context_enriched.csv"
AGE_WINDOWS = [(22, 36), (23, 35), (24, 34), (25, 33)]


def _season_year(value: object) -> float:
    try:
        return float(season_start_year(str(value)))
    except Exception:
        return float("nan")


def _load_main() -> pd.DataFrame:
    if not MAIN_INPUT.exists():
        raise FileNotFoundError(f"Missing main player-season input: {MAIN_INPUT}")
    df = pd.read_csv(MAIN_INPUT)
    df["season_start_year"] = pd.to_numeric(df.get("season_start_year"), errors="coerce")
    return df


def _add_prior_cba_features(df: pd.DataFrame) -> pd.DataFrame:
    label_path = PROCESSED_DIR / "cba_imports_extended_verified.csv"
    if not label_path.exists():
        label_path = PROCESSED_DIR / "cba_imports_extended.csv"
    labels = pd.read_csv(label_path)
    labels["cba_start_year"] = labels["cba_season"].map(_season_year)
    by_player = {
        key: group.sort_values("cba_start_year")
        for key, group in labels.dropna(subset=["player_name_key", "cba_start_year"]).groupby("player_name_key")
    }
    rows = []
    for row in df.itertuples(index=False):
        player_labels = by_player.get(str(row.player_name_key))
        t = getattr(row, "season_start_year")
        if player_labels is None or pd.isna(t):
            prior = pd.DataFrame()
        else:
            prior = player_labels[player_labels["cba_start_year"] < float(t)].copy()
        prior_seasons = sorted(prior["cba_season"].dropna().astype(str).unique()) if not prior.empty else []
        last_year = float(prior["cba_start_year"].max()) if not prior.empty else float("nan")
        rows.append(
            {
                "has_prior_cba_experience_before_t": int(bool(prior_seasons)),
                "prior_cba_seasons_before_t": len(prior_seasons),
                "prior_cba_total_appearances_before_t": len(prior),
                "prior_cba_last_seen_season": prior.loc[prior["cba_start_year"].idxmax(), "cba_season"] if not prior.empty else pd.NA,
                "prior_cba_last_seen_gap": float(t - last_year) if not prior.empty and pd.notna(t) else pd.NA,
                "prior_cba_is_returning_import": int(not prior.empty and pd.notna(t) and (t - last_year) <= 3),
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _age_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_available"] = False
    out["age_at_season_start"] = pd.NA
    out["age_source_id"] = pd.NA
    for col in ["age_at_season_start", "age_at_season", "age"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            out["age_at_season_start"] = pd.to_numeric(df[col], errors="coerce")
            out["age_available"] = out["age_at_season_start"].notna()
            out["age_source_id"] = out.get("source_id", "unknown")
            return out
    for col in ["birth_year"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            out["age_at_season_start"] = out["season_start_year"] - pd.to_numeric(df[col], errors="coerce")
            out["age_available"] = out["age_at_season_start"].notna()
            out["age_source_id"] = out.get("source_id", "unknown")
            return out
    return out


def _add_trends(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_name_key", "season_start_year"]).copy()
    base_cols = [
        "points_per_36",
        "usage_proxy",
        "ts_pct",
        "minutes_per_game",
        "assists_per_36",
        "rebounds_per_36",
        "turnovers_per_36",
    ]
    for col in base_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
        prev = out.groupby("player_name_key")[col].shift(1)
        prev_year = out.groupby("player_name_key")["season_start_year"].shift(1)
        valid_prev = prev.notna() & prev_year.lt(out["season_start_year"])
        out[f"{col}_previous"] = prev.where(valid_prev)
        out[f"{col}_trend"] = (out[col] - prev).where(valid_prev)
        out[f"{col}_trend_missing"] = (~valid_prev).astype(int)
    trend_cols = [f"{col}_trend" for col in base_cols]
    out["has_previous_season_record"] = out.groupby("player_name_key")["season_start_year"].shift(1).lt(out["season_start_year"]).fillna(False).astype(int)
    out["trend_features_available_count"] = out[trend_cols].notna().sum(axis=1)
    return out.sort_index()


def _write_prior_report(df: pd.DataFrame) -> None:
    prior = df["has_prior_cba_experience_before_t"].eq(1)
    rows = [
        {"metric": "total_rows", "value": len(df)},
        {"metric": "rows_with_prior_cba_experience", "value": int(prior.sum())},
        {"metric": "positives_with_prior_cba_experience", "value": int(df.loc[prior, "signed_cba_next_season"].sum())},
        {"metric": "positive_rate_with_prior_cba_experience", "value": df.loc[prior, "signed_cba_next_season"].mean() if prior.any() else 0},
        {"metric": "positive_rate_without_prior_cba_experience", "value": df.loc[~prior, "signed_cba_next_season"].mean() if (~prior).any() else 0},
    ]
    summary = pd.DataFrame(rows)
    by_season = df.groupby("season", dropna=False).agg(
        rows=("player_name_key", "size"),
        rows_with_prior_cba=("has_prior_cba_experience_before_t", "sum"),
        positives=("signed_cba_next_season", "sum"),
    ).reset_index()
    examples = df[prior].sort_values(["signed_cba_next_season", "prior_cba_seasons_before_t"], ascending=False).head(30)
    pd.concat(
        [
            summary.assign(section="overall"),
            by_season.rename(columns={"season": "metric"}).assign(section="by_season", value=pd.NA),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(REPORTS_DIR / "prior_cba_feature_summary.csv", index=False)
    examples[
        [
            "player_name_raw",
            "season",
            "next_season",
            "signed_cba_next_season",
            "prior_cba_seasons_before_t",
            "prior_cba_last_seen_season",
            "prior_cba_last_seen_gap",
        ]
    ].to_csv(REPORTS_DIR / "prior_cba_repeat_import_examples.csv", index=False)


def _write_age_reports(df: pd.DataFrame) -> None:
    age_available = df["age_available"].fillna(False).astype(bool)
    source_summary = df.groupby("source_id", dropna=False).agg(
        rows=("player_name_key", "size"),
        age_available_rows=("age_available", "sum"),
        positives=("signed_cba_next_season", "sum"),
    ).reset_index()
    source_summary["age_coverage"] = source_summary["age_available_rows"] / source_summary["rows"]
    overall = pd.DataFrame(
        [
            {"source_id": "ALL", "rows": len(df), "age_available_rows": int(age_available.sum()), "positives": int(df["signed_cba_next_season"].sum()), "age_coverage": age_available.mean()},
        ]
    )
    pd.concat([overall, source_summary], ignore_index=True).to_csv(REPORTS_DIR / "age_distribution_summary.csv", index=False)
    rows = []
    for low, high in AGE_WINDOWS:
        if age_available.any():
            mask = df["age_at_season_start"].between(low, high, inclusive="both")
            pool = df[mask].copy()
        else:
            mask = pd.Series(False, index=df.index)
            pool = df.iloc[0:0].copy()
        pool.to_csv(PROCESSED_DIR / f"pool_age_{low}_{high}.csv", index=False)
        rows.append(
            {
                "age_window": f"{low}_{high}",
                "candidate_rows_retained": int(mask.sum()),
                "positives_retained": int(df.loc[mask, "signed_cba_next_season"].sum()) if mask.any() else 0,
                "positive_rate": df.loc[mask, "signed_cba_next_season"].mean() if mask.any() else 0,
                "positive_retention_rate": df.loc[mask, "signed_cba_next_season"].sum() / df["signed_cba_next_season"].sum() if df["signed_cba_next_season"].sum() else 0,
                "candidate_reduction_rate": 1 - mask.mean(),
                "age_available": bool(age_available.any()),
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "age_gate_sensitivity.csv", index=False)


def _write_trend_report(df: pd.DataFrame) -> None:
    trend_cols = [c for c in df.columns if c.endswith("_trend")]
    rows = [{"metric": "rows", "value": len(df)}, {"metric": "rows_with_previous_season_record", "value": int(df["has_previous_season_record"].sum())}]
    for col in trend_cols:
        rows.append({"metric": f"{col}_non_null_rows", "value": int(df[col].notna().sum())})
    rows.append({"metric": "mean_trend_features_available_count", "value": df["trend_features_available_count"].mean()})
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "trend_feature_summary.csv", index=False)


def run() -> None:
    ensure_data_dirs()
    df = _load_main()
    df = _add_prior_cba_features(df)
    df = _age_candidates(df)
    df = _add_trends(df)
    df.to_csv(OUTPUT, index=False)
    _write_prior_report(df)
    _write_age_reports(df)
    _write_trend_report(df)
    print(f"Wrote context-enriched dataset: {OUTPUT}")


if __name__ == "__main__":
    run()
