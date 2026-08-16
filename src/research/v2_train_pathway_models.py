from __future__ import annotations

import importlib.util
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import player_season_rank_utils as ps
from .v2_pathway_utils import V2_PROCESSED_DIR, V2_REPORTS_DIR, ensure_v2_dirs


POOLS = {
    "broad_eligible_overseas_pool": V2_PROCESSED_DIR / "pool_broad_eligible.csv",
    "common_cba_source_league_pool": V2_PROCESSED_DIR / "pool_common_cba_source.csv",
    "expanded_pathway_pool": V2_PROCESSED_DIR / "pool_expanded_pathway.csv",
    "australian_nbl_augmented_pool": V2_PROCESSED_DIR / "pool_australian_nbl_augmented.csv",
    "career_pathway_signal_pool": V2_PROCESSED_DIR / "pool_career_pathway_signal.csv",
}
LABELS = ["signed_cba_next_season", "signed_cba_within_2_seasons", "signed_cba_within_3_seasons"]
TOP_K = [20, 50, 100, 300]

RAW_FEATURES = ["minutes_per_game", "points_per_36", "usage_proxy", "ts_pct", "cba_import_fit_score"]
PATHWAY_FEATURES = [
    "number_of_prior_overseas_leagues",
    "number_of_prior_overseas_seasons",
    "has_gleague_experience_before_t",
    "has_euroleague_experience_before_t",
    "has_eurocup_experience_before_t",
    "has_australian_nbl_experience_before_t",
    "has_japanese_bleague_experience_before_t",
    "has_korean_kbl_experience_before_t",
    "has_prior_cba_experience_before_t",
    "australian_nbl_seasons_before_t",
    "australian_nbl_last_seen_gap",
    "australian_nbl_last_points_per_36",
    "australian_nbl_last_usage_proxy",
    "australian_nbl_last_ts_pct",
    "australian_nbl_is_immediate_previous_league",
    "australian_nbl_any_experience_before_t",
    "australian_nbl_current_or_prior_experience_to_t",
]
LEAGUE_ADJUSTED_CANDIDATES = [
    "minutes_per_game_league_z",
    "points_per_36_league_z",
    "usage_proxy_league_z",
    "ts_pct_league_z",
    "league_season_minutes_z",
    "league_season_points_z",
    "league_season_usage_z",
    "league_season_ts_z",
]


# 函数：_feature_sets
def _feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    raw = [c for c in RAW_FEATURES if c in df.columns]
    pathway = [c for c in PATHWAY_FEATURES if c in df.columns]
    adjusted = [c for c in LEAGUE_ADJUSTED_CANDIDATES if c in df.columns]
    return {
        "raw_performance": raw,
        "career_pathway": pathway,
        "raw_plus_pathway": raw + pathway,
        "league_adjusted": adjusted or raw,
        "raw_plus_league_adjusted_plus_pathway": list(dict.fromkeys(raw + adjusted + pathway)),
    }


# 函数：_metric
def _metric(y: pd.Series, score: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(score, errors="coerce").fillna(-999)
    ranked = pd.DataFrame({"y": y, "score": score}).sort_values("score", ascending=False).reset_index(drop=True)
    pos = int(ranked["y"].sum())
    base = pos / len(ranked) if len(ranked) else 0.0
    hits_idx = ranked.index[ranked["y"].eq(1)].tolist()
    out = {
        "candidate_count": len(ranked),
        "positive_count": pos,
        "base_positive_rate": base,
        "pr_auc": float(average_precision_score(y, score)) if pos and y.nunique() > 1 else 0.0,
        "mrr": float(1 / (hits_idx[0] + 1)) if hits_idx else 0.0,
        "first_hit_rank": int(hits_idx[0] + 1) if hits_idx else pd.NA,
        "ndcg_at_20": float(ndcg_score([y.to_numpy()], [score.to_numpy()], k=min(20, len(y)))) if pos and len(y) else 0.0,
        "ndcg_at_100": float(ndcg_score([y.to_numpy()], [score.to_numpy()], k=min(100, len(y)))) if pos and len(y) else 0.0,
    }
    for k in TOP_K:
        top = ranked.head(k)
        hits = int(top["y"].sum())
        precision = hits / len(top) if len(top) else 0.0
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = hits / pos if pos else 0.0
        out[f"lift_at_{k}"] = precision / base if base else 0.0
        out[f"hit_count_at_{k}"] = hits
    return out


# 函数：_prep_model
def _prep_model(train: pd.DataFrame, features: list[str]) -> Pipeline:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) or pd.to_numeric(train[c], errors="coerce").notna().any()]
    categorical = [c for c in features if c not in numeric]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])


# 函数：_logistic_score
def _logistic_score(train: pd.DataFrame, test: pd.DataFrame, label: str, features: list[str]) -> pd.Series | None:
    if train[label].nunique() < 2 or not features:
        return None
    model = _prep_model(train, features)
    model.fit(train[features], train[label])
    return pd.Series(model.predict_proba(test[features])[:, 1], index=test.index)


# 函数：_lgbm_rank_score
def _lgbm_rank_score(train: pd.DataFrame, test: pd.DataFrame, label: str, features: list[str]) -> pd.Series | None:
    if importlib.util.find_spec("lightgbm") is None or train[label].nunique() < 2 or not features:
        return None
    import lightgbm as lgb

    numeric_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(train[features].apply(pd.to_numeric, errors="coerce").median()).fillna(0)
    numeric_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(numeric_train.median()).fillna(0)
    groups = train.sort_values("season_start_year").groupby("season_start_year").size().to_list()
    train_sorted = train.sort_values("season_start_year")
    x_sorted = numeric_train.loc[train_sorted.index]
    y_sorted = train_sorted[label].astype(int)
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=80, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(x_sorted, y_sorted, group=groups)
    return pd.Series(model.predict(numeric_test), index=test.index)


# 函数：_run_pool
def _run_pool(pool_name: str, df: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    results, preds, feature_rows = [], [], []
    years = sorted(pd.to_numeric(df["season_start_year"], errors="coerce").dropna().astype(int).unique())
    fsets = _feature_sets(df)
    for label in LABELS:
        for feature_set, features in fsets.items():
            feature_rows.append({"pool": pool_name, "label": label, "feature_set": feature_set, "features": "; ".join(features)})
        for year in years[1:]:
            train = df[pd.to_numeric(df["season_start_year"], errors="coerce") < year].copy()
            test = df[pd.to_numeric(df["season_start_year"], errors="coerce") == year].copy()
            if test.empty:
                continue
            if int(test[label].sum()) == 0 or int(train[label].sum()) == 0:
                continue
            model_scores: dict[str, pd.Series] = {"transparent_rule_based": ps.fit_rule_score(test)}
            for feature_set, features in fsets.items():
                log_score = _logistic_score(train, test, label, features)
                if log_score is not None:
                    model_scores[f"logistic_balanced__{feature_set}"] = log_score
                if feature_set in {"raw_plus_pathway", "raw_plus_league_adjusted_plus_pathway"}:
                    lgb_score = _lgbm_rank_score(train, test, label, features)
                    if lgb_score is not None:
                        model_scores[f"lightgbm_lambdarank__{feature_set}"] = lgb_score
            for model_name, score in model_scores.items():
                meta = {"pool": pool_name, "label": label, "model": model_name, "test_year": year, "test_season": str(test["season"].iloc[0])}
                row = dict(meta)
                row.update(_metric(test[label], score))
                row["train_rows"] = len(train)
                row["train_positive_count"] = int(train[label].sum())
                results.append(row)
                ranked = test.copy()
                ranked["v2_score"] = pd.to_numeric(score, errors="coerce").fillna(-999)
                ranked["v2_rank"] = ranked["v2_score"].rank(method="first", ascending=False).astype(int)
                for k, v in meta.items():
                    ranked[k] = v
                preds.append(ranked.sort_values("v2_rank").head(300))
    return results, preds, feature_rows


# 函数：run
def run() -> None:
    ensure_v2_dirs()
    all_results, all_preds, all_features = [], [], []
    for pool_name, path in POOLS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        results, preds, feature_rows = _run_pool(pool_name, df)
        all_results.extend(results)
        all_preds.extend(preds)
        all_features.extend(feature_rows)
    results_df = pd.DataFrame(all_results)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    features_df = pd.DataFrame(all_features).drop_duplicates()
    results_df.to_csv(V2_REPORTS_DIR / "v2_model_results.csv", index=False)
    preds_df.to_csv(V2_REPORTS_DIR / "v2_topk_predictions.csv", index=False)
    features_df.to_csv(V2_REPORTS_DIR / "v2_model_feature_sets.csv", index=False)

    aus = results_df[results_df["pool"].str.contains("australian", case=False, na=False) | results_df["pool"].eq("common_cba_source_league_pool")]
    aus.to_csv(V2_REPORTS_DIR / "v2_australian_nbl_model_comparison.csv", index=False)
    print(f"Wrote v2 model results to {V2_REPORTS_DIR / 'v2_model_results.csv'}")
    if not results_df.empty:
        best = results_df[results_df["label"].eq("signed_cba_next_season")].sort_values(["precision_at_20", "precision_at_50", "recall_at_100"], ascending=False).head(10)
        print(best[["pool", "label", "model", "precision_at_20", "precision_at_50", "recall_at_100", "recall_at_300", "lift_at_20"]].to_string(index=False))


if __name__ == "__main__":
    run()
