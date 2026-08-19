from __future__ import annotations

import pandas as pd

from .v2_pathway_utils import V2_REPORTS_DIR, ensure_v2_dirs


BASELINE = {
    "precision_at_20": 0.0464,
    "precision_at_50": 0.0429,
    "recall_at_100": 0.4832,
    "recall_at_300": 0.6818,
    "lift_at_20": 11.36,
}


def _fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def run() -> None:
    ensure_v2_dirs()
    audit = pd.read_csv(V2_REPORTS_DIR / "league_pathway_audit.csv")
    pools = pd.read_csv(V2_REPORTS_DIR / "candidate_pool_comparison.csv")
    results = pd.read_csv(V2_REPORTS_DIR / "v2_model_results.csv")
    leakage = pd.read_csv(V2_REPORTS_DIR / "v2_pathway_leakage_audit.csv")

    aus = audit[audit["league"].astype(str).str.contains("Australian", case=False, na=False)]
    aus_rows = int(aus["total_player_season_rows"].sum()) if not aus.empty else 0
    aus_players = int(aus["unique_players"].sum()) if not aus.empty else 0
    aus_next = int(aus["next_season_cba_positives"].sum()) if not aus.empty else 0
    aus_w2 = int(aus["within_2_seasons_count"].sum()) if not aus.empty else 0
    aus_w3 = int(aus["within_3_seasons_count"].sum()) if not aus.empty else 0
    aus_w5 = int(aus["within_5_seasons_count"].sum()) if not aus.empty else 0
    aus_any = int(aus["any_future_cba_count"].sum()) if not aus.empty else 0
    avg_gap = aus["average_gap_to_first_later_cba"].dropna().mean() if not aus.empty else pd.NA
    med_gap = aus["median_gap_to_first_later_cba"].dropna().median() if not aus.empty else pd.NA

    metric_cols = [
        "precision_at_20",
        "precision_at_50",
        "precision_at_100",
        "precision_at_300",
        "recall_at_100",
        "recall_at_300",
        "lift_at_20",
        "mrr",
        "ndcg_at_20",
        "ndcg_at_100",
    ]
    grouped = (
        results.groupby(["pool", "label", "model"], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped["test_splits"] = results.groupby(["pool", "label", "model"], dropna=False).size().to_numpy()
    grouped.to_csv(V2_REPORTS_DIR / "v2_model_results_aggregated.csv", index=False)
    next_results = grouped[grouped["label"].eq("signed_cba_next_season")].copy()
    best = next_results.sort_values(["precision_at_20", "precision_at_50", "recall_at_100"], ascending=False).iloc[0] if not next_results.empty else pd.Series()
    common = next_results[next_results["pool"].eq("common_cba_source_league_pool")].sort_values(["precision_at_20", "precision_at_50"], ascending=False).head(1)
    aus_pool = next_results[next_results["pool"].eq("australian_nbl_augmented_pool")].sort_values(["precision_at_20", "precision_at_50"], ascending=False).head(1)
    leak_pass = bool(leakage["passed"].astype(int).all())

    aus_immediate_rate = aus_next / aus_rows if aus_rows else 0
    aus_longer_signal = aus_w3 > aus_next or aus_w5 > aus_next
    aus_model_improves = False
    if not common.empty and not aus_pool.empty:
        aus_model_improves = float(aus_pool["precision_at_20"].iloc[0]) > float(common["precision_at_20"].iloc[0]) or float(aus_pool["recall_at_100"].iloc[0]) > float(common["recall_at_100"].iloc[0])

    lines = [
        "# V2 Pathway Rebuild Summary",
        "",
        "## 1. Purpose",
        "",
        "This v2 experiment rebuilds pathway labels and career-history features without overwriting the original dissertation outputs. The original final results remain the baseline reference.",
        "",
        "## 2. Australian-NBL Pathway Audit",
        "",
        f"- Australian-NBL rows: **{aus_rows}**",
        f"- Australian-NBL unique players: **{aus_players}**",
        f"- Direct next-season CBA positives: **{aus_next}**",
        f"- Next-season positive rate: **{_fmt(aus_immediate_rate, 6)}**",
        f"- Within 2 seasons positives: **{aus_w2}**",
        f"- Within 3 seasons positives: **{aus_w3}**",
        f"- Within 5 seasons positives: **{aus_w5}**",
        f"- Any later CBA count: **{aus_any}**",
        f"- Average gap to first later CBA: **{_fmt(avg_gap, 2)}**",
        f"- Median gap to first later CBA: **{_fmt(med_gap, 2)}**",
        "",
        "Interpretation: Australian-NBL is weak as an immediate next-season source if the direct positive count is low. If within-3 or within-5 counts exceed next-season positives, it is better treated as a longer-term career pathway signal rather than a direct transition label.",
        "",
        "## 3. Candidate Pool Comparison",
        "",
        "```text",
        pools.to_string(index=False),
        "```",
        "",
        "## 4. V2 Best Next-Season Ranking Result",
        "",
        "The comparison below uses mean metrics across valid walk-forward test splits, not a single best season.",
        "",
        f"- Best pool: **{best.get('pool', 'NA')}**",
        f"- Best model: **{best.get('model', 'NA')}**",
        f"- Precision@20: **{_fmt(best.get('precision_at_20', pd.NA))}**",
        f"- Precision@50: **{_fmt(best.get('precision_at_50', pd.NA))}**",
        f"- Recall@100: **{_fmt(best.get('recall_at_100', pd.NA))}**",
        f"- Recall@300: **{_fmt(best.get('recall_at_300', pd.NA))}**",
        f"- Lift@20: **{_fmt(best.get('lift_at_20', pd.NA))}**",
        "",
        "Original stable baseline:",
        "",
        f"- Precision@20: **{BASELINE['precision_at_20']}**",
        f"- Precision@50: **{BASELINE['precision_at_50']}**",
        f"- Recall@100: **{BASELINE['recall_at_100']}**",
        f"- Recall@300: **{BASELINE['recall_at_300']}**",
        f"- Lift@20: **{BASELINE['lift_at_20']}**",
        "",
        "## 5. Australian-NBL Model Impact",
        "",
        f"- Australian-NBL longer-window pathway signal observed: **{aus_longer_signal}**",
        f"- Australian-NBL augmented pool improves at least one key metric over common pool in this run: **{aus_model_improves}**",
        "",
        "Australian-NBL should not be included automatically unless the v2 model comparison shows a clear benefit. If it mainly increases within-3/5 coverage but hurts top precision, it should be discussed as a career-pathway/context signal rather than a replacement for the final recommendation logic.",
        "",
        "## 6. Leakage Audit",
        "",
        f"- Leakage audit all passed: **{leak_pass}**",
        "",
        "## 7. Claim Boundary",
        "",
        "Can claim: v2 tests whether longer-window pathway labels reveal career-history signals that immediate t -> t+1 labels miss.",
        "",
        "Cannot claim: Australian-NBL is important without evidence, or that v2 should replace the original dissertation result unless Top-K metrics and leakage checks support it.",
    ]
    out = V2_REPORTS_DIR / "v2_pathway_rebuild_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote v2 summary to {out}")


if __name__ == "__main__":
    run()
