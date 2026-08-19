from __future__ import annotations

from io import StringIO

import pandas as pd

from .source_diagnostics_utils import add_source_group
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


def run() -> None:
    ensure_data_dirs()
    ps = add_source_group(pd.read_csv(PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"))
    watch = pd.read_csv(REPORTS_DIR / "final_player_season_watchlist_gleague.csv")
    watch = add_source_group(watch)
    merged = ps.merge(
        watch[["player_name_key", "season", "rank", "model_score", "source_group"]],
        on=["player_name_key", "season"],
        how="left",
        suffixes=("", "_watch"),
    )
    if "source_group_watch" in merged.columns:
        merged["source_group"] = merged["source_group_watch"].fillna(merged["source_group"])

    rows = []
    for group, df in merged.groupby("source_group", dropna=False):
        pos = df[df["signed_cba_next_season"].eq(1)]
        neg = df[df["signed_cba_next_season"].eq(0)]
        row = {
            "source_group": group,
            "candidate_rows": len(df),
            "positive_rows": int(df["signed_cba_next_season"].sum()),
            "positive_rate": df["signed_cba_next_season"].mean(),
            "top20_hits": int((pos["rank"].le(20)).sum()),
            "top50_hits": int((pos["rank"].le(50)).sum()),
            "top100_hits": int((pos["rank"].le(100)).sum()),
            "average_rank_of_positives": pd.to_numeric(pos["rank"], errors="coerce").mean(),
            "median_rank_of_positives": pd.to_numeric(pos["rank"], errors="coerce").median(),
            "mean_model_score": pd.to_numeric(df["model_score"], errors="coerce").mean(),
            "mean_score_for_positives": pd.to_numeric(pos["model_score"], errors="coerce").mean(),
            "mean_score_for_negatives": pd.to_numeric(neg["model_score"], errors="coerce").mean(),
        }
        row["score_separation_pos_minus_neg"] = row["mean_score_for_positives"] - row["mean_score_for_negatives"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("positive_rows", ascending=False)
    out.to_csv(REPORTS_DIR / "source_specific_performance.csv", index=False)

    top_source = out.sort_values("positive_rows", ascending=False).head(1)
    top_hits = out.sort_values("top100_hits", ascending=False).head(1)
    buffer = StringIO()
    out.to_csv(buffer, index=False)
    lines = [
        "# Source-Specific Performance",
        "",
        "```csv",
        buffer.getvalue().strip(),
        "```",
        "",
        f"- Source contributing most positives: {top_source.iloc[0]['source_group'] if not top_source.empty else 'n/a'}",
        f"- Source contributing most Top100 hits: {top_hits.iloc[0]['source_group'] if not top_hits.empty else 'n/a'}",
        "G League ranking dilution is indicated when G League positive rate is high but Top-K hits or score separation are weak relative to other sources.",
    ]
    (REPORTS_DIR / "source_specific_performance.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote source-specific performance report")


if __name__ == "__main__":
    run()
