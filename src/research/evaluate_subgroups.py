from __future__ import annotations

import pandas as pd

from . import large_scale_rank_utils as lu
from . import player_season_rank_utils as ps
from .ranking_metric_utils import ranked_predictions, ranking_metrics
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


# 函数：_load_scores
def _load_scores() -> pd.DataFrame:
    ltr = REPORTS_DIR / "ltr_topk_predictions.csv"
    if ltr.exists():
        return pd.read_csv(ltr)
    df = pd.read_csv(PROCESSED_DIR / "labelled_player_season_dataset_gleague_league_adjusted.csv")
    rows = []
    for year in ps.split_years(df):
        test = df[df["season_start_year"].eq(year)].copy()
        common = test[lu.pool_mask(test, "pool_common_cba_source_leagues")].copy()
        if common.empty:
            continue
        score = ps.fit_rule_score(common)
        rows.append(ranked_predictions(common, score, {"pool": "common_cba_source_leagues", "model": "rule_based", "feature_variant": "current_rule", "test_year": year}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# 函数：_subgroup_masks
def _subgroup_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {}
    text = (df.get("league", "").astype(str) + " " + df.get("leagues_played_that_season", "").astype(str)).str.lower()
    masks["g_league_candidates"] = text.str.contains("g league")
    masks["euroleague_eurocup_candidates"] = text.str.contains("euroleague|eurocup")
    masks["common_cba_source_league_pool"] = df.apply(lu.is_common_pathway, axis=1) if "league" in df.columns else pd.Series(False, index=df.index)
    masks["performance_only_pool"] = pd.Series(True, index=df.index)
    if "has_prior_cba_experience_before_t" in df.columns:
        masks["prior_cba_experience"] = df["has_prior_cba_experience_before_t"].fillna(0).astype(float).gt(0)
        masks["no_prior_cba_experience"] = ~masks["prior_cba_experience"]
    if "has_high_usage_row" in df.columns:
        masks["high_usage_subgroup"] = df["has_high_usage_row"].fillna(0).astype(float).gt(0)
    if "points_per_36" in df.columns:
        masks["scoring_role_subgroup"] = pd.to_numeric(df["points_per_36"], errors="coerce").ge(pd.to_numeric(df["points_per_36"], errors="coerce").median())
    return masks


# 函数：run
def run() -> None:
    ensure_data_dirs()
    pred = _load_scores()
    if pred.empty:
        pd.DataFrame().to_csv(REPORTS_DIR / "subgroup_ranking_evaluation.csv", index=False)
        return
    rows, mrr_rows = [], []
    for (model, variant, pool, year), group in pred.groupby(["model", "feature_variant", "pool", "test_year"], dropna=False):
        for name, mask in _subgroup_masks(group).items():
            sub = group[mask.reindex(group.index).fillna(False)]
            if sub.empty:
                continue
            metric = ranking_metrics(sub["signed_cba_next_season"], sub["score"])
            metric.update(
                {
                    "subgroup": name,
                    "model": model,
                    "feature_variant": variant,
                    "pool": pool,
                    "test_year": year,
                    "insufficient_sample_size": len(sub) < 20 or int(sub["signed_cba_next_season"].sum()) == 0,
                }
            )
            rows.append(metric)
            mrr_rows.append(
                {
                    "subgroup": name,
                    "model": model,
                    "feature_variant": variant,
                    "pool": pool,
                    "test_year": year,
                    "mrr": metric["mrr"],
                    "rank_of_first_true_positive": metric["rank_of_first_true_positive"],
                    "ndcg_at_20": metric["ndcg_at_20"],
                    "ndcg_at_100": metric["ndcg_at_100"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "subgroup_ranking_evaluation.csv", index=False)
    pd.DataFrame(mrr_rows).to_csv(REPORTS_DIR / "mrr_ndcg_summary.csv", index=False)
    if not out.empty:
        first = out.groupby(["subgroup", "model", "feature_variant"], as_index=False).agg(
            mean_first_hit_rank=("rank_of_first_true_positive", "mean"),
            mean_mrr=("mrr", "mean"),
            mean_ndcg_at_20=("ndcg_at_20", "mean"),
            mean_ndcg_at_100=("ndcg_at_100", "mean"),
            positive_rows=("test_positive_count", "sum"),
        ).sort_values(["mean_mrr", "mean_ndcg_at_20"], ascending=False)
    else:
        first = pd.DataFrame()
    first.to_csv(REPORTS_DIR / "first_hit_rank_summary.csv", index=False)
    print("Wrote subgroup ranking evaluation reports.")


if __name__ == "__main__":
    run()
