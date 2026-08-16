from __future__ import annotations

import pandas as pd

from .v2_pathway_utils import V2_PROCESSED_DIR, V2_REPORTS_DIR, ensure_v2_dirs, has_pathway, pool_mask


INPUT = V2_PROCESSED_DIR / "labelled_player_season_dataset_pathway_features.csv"
AUDIT = V2_REPORTS_DIR / "league_pathway_audit.csv"

POOLS = {
    "broad_eligible_overseas_pool": V2_PROCESSED_DIR / "pool_broad_eligible.csv",
    "common_cba_source_league_pool": V2_PROCESSED_DIR / "pool_common_cba_source.csv",
    "expanded_pathway_pool": V2_PROCESSED_DIR / "pool_expanded_pathway.csv",
    "australian_nbl_augmented_pool": V2_PROCESSED_DIR / "pool_australian_nbl_augmented.csv",
    "career_pathway_signal_pool": V2_PROCESSED_DIR / "pool_career_pathway_signal.csv",
}


# 函数：_expanded_leagues
def _expanded_leagues() -> set[str]:
    if not AUDIT.exists():
        return set()
    audit = pd.read_csv(AUDIT)
    keep = audit[
        (pd.to_numeric(audit["within_3_seasons_count"], errors="coerce").fillna(0) >= 2)
        | (pd.to_numeric(audit["within_5_seasons_count"], errors="coerce").fillna(0) >= 3)
        | (pd.to_numeric(audit["any_future_cba_count"], errors="coerce").fillna(0) >= 5)
    ]
    return set(keep["league"].dropna().astype(str))


# 函数：_pool_summary
def _pool_summary(name: str, pool: pd.DataFrame) -> dict[str, object]:
    return {
        "pool": name,
        "rows": len(pool),
        "unique_players": pool["player_name_key"].nunique(),
        "next_season_positives": int(pool["signed_cba_next_season"].sum()),
        "within_2_positives": int(pool["signed_cba_within_2_seasons"].sum()),
        "within_3_positives": int(pool["signed_cba_within_3_seasons"].sum()),
        "within_5_positives": int(pool["signed_cba_within_5_seasons"].sum()),
        "next_season_positive_rate": float(pool["signed_cba_next_season"].mean()) if len(pool) else 0.0,
        "within_3_positive_rate": float(pool["signed_cba_within_3_seasons"].mean()) if len(pool) else 0.0,
        "within_5_positive_rate": float(pool["signed_cba_within_5_seasons"].mean()) if len(pool) else 0.0,
    }


# 函数：run
def run() -> None:
    ensure_v2_dirs()
    df = pd.read_csv(INPUT)
    expanded_leagues = _expanded_leagues()
    df["expanded_pathway_flag"] = df["league"].astype(str).isin(expanded_leagues) | df.apply(lambda r: has_pathway(r, "australian_nbl"), axis=1)
    summaries = []
    league_dist_rows = []
    for name, path in POOLS.items():
        pool = df[pool_mask(df, name)].copy()
        pool.to_csv(path, index=False)
        summaries.append(_pool_summary(name, pool))
        dist = pool.groupby("league", dropna=False).agg(rows=("player_name_key", "size"), next_positives=("signed_cba_next_season", "sum"), within_5=("signed_cba_within_5_seasons", "sum")).reset_index()
        dist.insert(0, "pool", name)
        league_dist_rows.append(dist)
    summary = pd.DataFrame(summaries)
    summary.to_csv(V2_REPORTS_DIR / "candidate_pool_comparison.csv", index=False)
    pd.concat(league_dist_rows, ignore_index=True).to_csv(V2_REPORTS_DIR / "candidate_pool_league_distribution.csv", index=False)
    print(f"Wrote candidate pool comparison to {V2_REPORTS_DIR / 'candidate_pool_comparison.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
