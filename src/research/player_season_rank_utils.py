from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOP_K = [20, 50, 100, 200, 300]


# 函数：metric
def metric(y: pd.Series, score: pd.Series) -> dict[str, float]:
    score = pd.to_numeric(score, errors="coerce").fillna(-999)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    ranked = pd.DataFrame({"y": y, "score": score}).sort_values("score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    positives = int(ranked["y"].sum())
    base_rate = positives / n if n else 0.0
    out = {
        "candidate_count": n,
        "test_positive_count": positives,
        "base_positive_rate": base_rate,
        "pr_auc": float(average_precision_score(y, score)) if positives else 0.0,
        "mrr": 0.0,
    }
    hit_rows = ranked.index[ranked["y"].eq(1)].tolist()
    if hit_rows:
        out["mrr"] = 1 / (hit_rows[0] + 1)
    for k in TOP_K:
        top = ranked.head(k)
        hits = int(top["y"].sum())
        precision = hits / len(top) if len(top) else 0.0
        recall = hits / positives if positives else 0.0
        denominator = min(positives, k)
        cumulative_hits = top["y"].cumsum()
        average_precision = float(((cumulative_hits / np.arange(1, len(top) + 1)) * top["y"]).sum() / denominator) if denominator else 0.0
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"lift_at_{k}"] = precision / base_rate if base_rate else 0.0
        out[f"hit_count_at_{k}"] = hits
        out[f"map_at_{k}"] = average_precision
    return out


# 函数：split_years
def split_years(df: pd.DataFrame) -> list[int]:
    years = sorted(df.loc[df["signed_cba_next_season"].eq(1), "season_start_year"].dropna().unique())
    return [int(year) for year in years if df.loc[df["season_start_year"] < year, "signed_cba_next_season"].nunique() == 2]


# 函数：high_usage_pool
def high_usage_pool(df: pd.DataFrame) -> pd.DataFrame:
    median = df.groupby(["league", "season"], dropna=False)["usage_proxy"].transform("median")
    return df[(df["max_usage_proxy"] > median) | (df["has_high_usage_row"].fillna(0).astype(float).gt(0))].copy()


# 函数：time_aware_common_pool
def time_aware_common_pool(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = set(train.loc[train["signed_cba_next_season"] == 1, "league"].dropna().astype(str))
    train_out = train[train["league"].astype(str).isin(allowed) | train["leagues_played_that_season"].fillna("").apply(lambda x: any(l in str(x) for l in allowed))].copy()
    test_out = test[test["league"].astype(str).isin(allowed) | test["leagues_played_that_season"].fillna("").apply(lambda x: any(l in str(x) for l in allowed))].copy()
    return train_out, test_out


# 函数：features
def features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {
        "player_name_raw",
        "player_name_key",
        "season",
        "next_season",
        "signed_cba_next_season",
        "leagues_played_that_season",
        "teams_played_that_season",
        "sources_present",
        "best_underlying_row_id",
        "best_underlying_team",
        "best_row_team",
    }
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and pd.to_numeric(df[col], errors="coerce").notna().any():
            numeric.append(col)
        elif col in {"league", "source_id", "best_source_id", "best_row_league", "best_underlying_league", "best_row_role_cluster_label", "role_cluster_label"}:
            categorical.append(col)
    return numeric, categorical


# 函数：logistic_scores
def logistic_scores(train: pd.DataFrame, test: pd.DataFrame, sample_weight=None) -> pd.Series | None:
    if train.empty or test.empty or train["signed_cba_next_season"].nunique() < 2:
        return None
    numeric, categorical = features(train)
    feature_columns = numeric + categorical
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight.reindex(train.index).fillna(1).to_numpy()
    model.fit(train[feature_columns], train["signed_cba_next_season"], **fit_kwargs)
    return pd.Series(model.predict_proba(test[feature_columns])[:, 1], index=test.index)



# 函数：fit_rule_score
def fit_rule_score(df: pd.DataFrame) -> pd.Series:
    return ltr._rule_score(df.rename(columns={"cba_import_fit_score": "fit_score_tmp"})) if False else (
        pd.to_numeric(df.get("max_cba_import_fit_score", df.get("cba_import_fit_score")), errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(df.get("max_usage_proxy", df.get("usage_proxy")), errors="coerce").fillna(0)
        + 0.02 * pd.to_numeric(df.get("max_points_per_36", df.get("points_per_36")), errors="coerce").fillna(0)
    )


# 函数：evaluate_scores
def evaluate_scores(test: pd.DataFrame, score: pd.Series, meta: dict) -> tuple[dict, pd.DataFrame]:
    row = dict(meta)
    row.update(metric(test["signed_cba_next_season"], score))
    ranked = test.copy()
    ranked["score"] = pd.to_numeric(score, errors="coerce").fillna(-999).to_numpy()
    ranked["rank"] = ranked["score"].rank(method="first", ascending=False).astype(int)
    for key, value in meta.items():
        ranked[key] = value
    tp = ranked[ranked["signed_cba_next_season"] == 1].sort_values("rank").head(20)
    return row, tp
