from __future__ import annotations

import importlib.util

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import large_scale_rank_utils as lu
from . import player_season_rank_utils as ps
from .ranking_metric_utils import ranked_predictions, ranking_metrics
from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


INPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague_league_adjusted.csv"
SEED = 42
EXCLUDE = {
    "player_name_raw",
    "player_name_key",
    "season",
    "next_season",
    "signed_cba_next_season",
    "prior_cba_last_seen_season",
}


# 函数：_load
def _load() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError("Run python -m src.build_league_adjusted_features first.")
    df = pd.read_csv(INPUT)
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")
    return df


# 函数：_pool
def _pool(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "full_current_player_season":
        return df.copy()
    if name == "common_cba_source_leagues":
        return df[lu.pool_mask(df, "pool_common_cba_source_leagues")].copy()
    if name == "performance_only":
        return df.copy()
    if name == "context_enriched":
        return df.copy()
    raise ValueError(name)


# 函数：_feature_cols
def _feature_cols(df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    all_numeric, all_cat = ps.features(df)
    league_adjusted = [c for c in all_numeric if "_league_season_" in c or c in {"league_season_group_size", "league_season_low_sample_flag"}]
    context_cols = [
        c
        for c in all_numeric
        if c.startswith("prior_cba_")
        or c.startswith("has_prior_cba")
        or c.startswith("trend_")
        or c.endswith("_trend")
        or c.endswith("_trend_missing")
        or c in {"has_previous_season_record", "trend_features_available_count"}
    ]
    if variant == "raw_features":
        numeric = [c for c in all_numeric if c not in league_adjusted and c not in context_cols]
    elif variant == "league_adjusted_features":
        numeric = league_adjusted
    elif variant == "raw_plus_league_adjusted_plus_context":
        numeric = all_numeric
    else:
        numeric = all_numeric
    numeric = [c for c in numeric if c not in EXCLUDE and pd.to_numeric(df[c], errors="coerce").notna().any()]
    categorical = [c for c in all_cat if c not in EXCLUDE and c in df.columns]
    return numeric, categorical


# 函数：_preprocess
def _preprocess(train: pd.DataFrame, test: pd.DataFrame, variant: str):
    numeric, categorical = _feature_cols(train, variant)
    features = numeric + categorical
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    train_x = pre.fit_transform(train[features])
    test_x = pre.transform(test[features])
    return train_x, test_x, features


# 函数：_group_sizes
def _group_sizes(train: pd.DataFrame) -> list[int]:
    sorted_train = train.sort_values("season_start_year")
    return [len(group) for _, group in sorted_train.groupby("season_start_year", sort=True)]


# 函数：_lgbm_scores
def _lgbm_scores(train: pd.DataFrame, test: pd.DataFrame, variant: str) -> tuple[pd.Series | None, list[str]]:
    if importlib.util.find_spec("lightgbm") is None:
        return None, []
    from lightgbm import LGBMRanker

    if train.empty or test.empty or train["signed_cba_next_season"].nunique() < 2:
        return None, []
    train_sorted = train.sort_values("season_start_year").copy()
    train_x, test_x, features = _preprocess(train_sorted, test, variant)
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=15,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(train_x, train_sorted["signed_cba_next_season"], group=_group_sizes(train_sorted))
    return pd.Series(model.predict(test_x), index=test.index), features


# 函数：_logistic_scores
def _logistic_scores(train: pd.DataFrame, test: pd.DataFrame, variant: str) -> tuple[pd.Series | None, list[str]]:
    if train.empty or test.empty or train["signed_cba_next_season"].nunique() < 2:
        return None, []
    train_x, test_x, features = _preprocess(train, test, variant)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEED)
    model.fit(train_x, train["signed_cba_next_season"])
    return pd.Series(model.predict_proba(test_x)[:, 1], index=test.index), features


# 函数：run
def run() -> None:
    ensure_data_dirs()
    df = _load()
    lightgbm_available = importlib.util.find_spec("lightgbm") is not None
    pools = ["full_current_player_season", "common_cba_source_leagues", "performance_only", "context_enriched"]
    variants = ["raw_features", "league_adjusted_features", "raw_plus_league_adjusted_plus_context"]
    results, preds, used_features = [], [], {}
    years = ps.split_years(df)
    for pool_name in pools:
        pool_df = _pool(df, pool_name)
        for year in years:
            train = pool_df[pool_df["season_start_year"] < year].copy()
            test = pool_df[pool_df["season_start_year"] == year].copy()
            if train.empty or test.empty:
                continue
            base_score = ps.fit_rule_score(test)
            meta = {"pool": pool_name, "model": "rule_based", "feature_variant": "current_rule", "test_year": year, "query_group": "candidate_season"}
            row = dict(meta)
            row.update(ranking_metrics(test["signed_cba_next_season"], base_score))
            row["train_groups"] = train["season_start_year"].nunique()
            row["test_groups"] = 1
            results.append(row)
            preds.append(ranked_predictions(test, base_score, meta))
            for variant in variants:
                score, features = _logistic_scores(train, test, variant)
                if score is not None:
                    meta = {"pool": pool_name, "model": "logistic_regression_balanced", "feature_variant": variant, "test_year": year, "query_group": "candidate_season"}
                    row = dict(meta)
                    row.update(ranking_metrics(test["signed_cba_next_season"], score))
                    row["train_groups"] = train["season_start_year"].nunique()
                    row["test_groups"] = 1
                    results.append(row)
                    preds.append(ranked_predictions(test, score, meta))
                    used_features[f"logistic_regression_balanced|{variant}"] = features
                score, features = _lgbm_scores(train, test, variant)
                if score is not None:
                    meta = {"pool": pool_name, "model": "lightgbm_lambdarank", "feature_variant": variant, "test_year": year, "query_group": "candidate_season"}
                    row = dict(meta)
                    row.update(ranking_metrics(test["signed_cba_next_season"], score))
                    row["train_groups"] = train["season_start_year"].nunique()
                    row["test_groups"] = 1
                    results.append(row)
                    preds.append(ranked_predictions(test, score, meta))
                    used_features[f"lightgbm_lambdarank|{variant}"] = features
    res = pd.DataFrame(results)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    res.to_csv(REPORTS_DIR / "ltr_model_results.csv", index=False)
    pred[pred["rank"] <= 300].to_csv(REPORTS_DIR / "ltr_topk_predictions.csv", index=False)
    lines = []
    for name, cols in used_features.items():
        lines.append(f"## {name}")
        lines.extend(cols)
        lines.append("")
    (REPORTS_DIR / "ltr_feature_columns_used.txt").write_text("\n".join(lines), encoding="utf-8")
    note = [
        "# LTR Install Or Skip Notes",
        "",
        f"LightGBM available: {lightgbm_available}",
        "",
        "If LightGBM is unavailable, install with:",
        "",
        "```powershell",
        "python -m pip install lightgbm",
        "```",
    ]
    if not res.empty:
        best_p20 = res.sort_values(["precision_at_20", "precision_at_50"], ascending=False).iloc[0]
        best_recall = res.sort_values("recall_at_100", ascending=False).iloc[0]
        note.extend(
            [
                "",
                f"Best Precision@20: {best_p20['precision_at_20']:.4f} ({best_p20['pool']} / {best_p20['model']} / {best_p20['feature_variant']})",
                f"Best Recall@100: {best_recall['recall_at_100']:.4f} ({best_recall['pool']} / {best_recall['model']} / {best_recall['feature_variant']})",
            ]
        )
    (REPORTS_DIR / "ltr_install_or_skip_notes.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print("Wrote LTR ranking experiment reports.")


if __name__ == "__main__":
    run()
