from __future__ import annotations

from io import StringIO

import pandas as pd

from . import large_scale_rank_utils as lu
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


POOLS = [
    "pool_high_confidence_pathways",
    "pool_common_cba_source_leagues",
    "pool_all_eligible_overseas",
    "pool_full_multisource",
]


# 函数：_summary
def _summary(df: pd.DataFrame, row_df: pd.DataFrame, pool_name: str) -> dict[str, object]:
    row_count = len(row_df[row_df["candidate_id"].isin(set(df["candidate_id"]))]) if "candidate_id" in row_df else pd.NA
    return {
        "pool": pool_name,
        "row_level_rows": row_count,
        "player_season_rows": len(df),
        "positives": int(df["signed_cba_next_season"].sum()),
        "positive_rate": df["signed_cba_next_season"].mean() if len(df) else 0,
        "seasons_covered": "; ".join(sorted(df["season"].dropna().astype(str).unique())),
        "sources_covered": "; ".join(sorted(df["source_group"].dropna().astype(str).unique())),
        "leagues_covered_count": df["league"].dropna().astype(str).nunique(),
        "positives_by_season": df[df["signed_cba_next_season"].eq(1)].groupby("next_season").size().to_json(),
        "positives_by_source": df[df["signed_cba_next_season"].eq(1)].groupby("source_group").size().to_json(),
        "positives_by_league": df[df["signed_cba_next_season"].eq(1)].groupby("league").size().sort_values(ascending=False).head(20).to_json(),
    }


# 函数：run
def run() -> None:
    ensure_data_dirs()
    ps_path = PROCESSED_DIR / "labelled_player_season_dataset_domestic.csv"
    row_path = PROCESSED_DIR / "labelled_candidate_dataset_multisource_domestic.csv"
    is_domestic_run = ps_path.exists() and row_path.exists()
    if not is_domestic_run:
        ps_path = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
        row_path = PROCESSED_DIR / "labelled_candidate_dataset_multisource_gleague.csv"
    ps = lu.load_ps(ps_path)
    row = pd.read_csv(row_path)
    row["candidate_id"] = [lu.stable_id(r.player_name_key, r.season) for r in row.itertuples(index=False)]
    summaries = []
    for pool in POOLS:
        subset = ps[lu.pool_mask(ps, pool)].copy()
        subset.to_csv(PROCESSED_DIR / f"{pool}{'_domestic' if is_domestic_run else ''}.csv", index=False)
        summaries.append(_summary(subset, row, pool))
    comp = pd.DataFrame(summaries)
    comp_name = "candidate_universe_comparison_domestic.csv" if is_domestic_run else "candidate_universe_comparison.csv"
    comp.to_csv(REPORTS_DIR / comp_name, index=False)
    buffer = StringIO()
    comp.drop(columns=["positives_by_season", "positives_by_source", "positives_by_league"]).to_csv(buffer, index=False)
    lines = [
        "# Candidate Universe Comparison",
        "",
        "```csv",
        buffer.getvalue().strip(),
        "```",
        "",
        "High-confidence pathways use prior-season positive pathway information only, so rows in early seasons are naturally limited.",
    ]
    md_name = "candidate_universe_comparison_domestic.md" if is_domestic_run else "candidate_universe_comparison.md"
    (REPORTS_DIR / md_name).write_text("\n".join(lines), encoding="utf-8")
    print("Wrote candidate universe variants")


if __name__ == "__main__":
    run()
