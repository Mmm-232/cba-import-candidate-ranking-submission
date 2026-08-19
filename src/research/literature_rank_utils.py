from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TOP_K = [10, 20, 50, 100]

LEAKAGE_COLUMNS = {
    "player_name_raw",
    "player_name_clean",
    "player_name_key",
    "next_season",
    "signed_cba_next_season",
    "cba_season",
    "source_note",
    "source_url",
    "source_url_or_file",
    "verification_status",
    "match_method",
    "match_confidence",
    "team",
    "source",
    "source_name",
    "extraction_date",
    "evidence_source_name",
    "evidence_url",
    "evidence_note",
}


def load_best_dataset(processed_dir: Path) -> pd.DataFrame:
    verified = processed_dir / "labelled_candidate_dataset_multisource_verified.csv"
    fallback = processed_dir / "labelled_candidate_dataset_multisource.csv"
    path = verified if verified.exists() else fallback
    df = pd.read_csv(path)
    return ensure_model_columns(df)


def ensure_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "season_start_year" not in df.columns:
        df["season_start_year"] = df["season"].astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)
    for col in [
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
        "points_per_36",
        "rebounds_per_36",
        "assists_per_36",
        "steals_per_36",
        "blocks_per_36",
        "turnovers_per_36",
        "shot_attempts_per_36",
        "three_point_attempt_rate",
        "free_throw_attempt_rate",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    games = df["games"].replace(0, np.nan)
    minutes = df["minutes"].replace(0, np.nan)
    if df["minutes_per_game"].isna().all():
        df["minutes_per_game"] = df["minutes"] / games
    df["points_per_game"] = df.get("points_per_game", df["points"] / games)
    df["points_per_36"] = df["points_per_36"].fillna(df["points"] / minutes * 36)
    df["rebounds_per_36"] = df["rebounds_per_36"].fillna(df["rebounds"] / minutes * 36)
    df["assists_per_36"] = df["assists_per_36"].fillna(df["assists"] / minutes * 36)
    df["steals_per_36"] = df["steals_per_36"].fillna(df["steals"] / minutes * 36)
    df["blocks_per_36"] = df["blocks_per_36"].fillna(df["blocks"] / minutes * 36)
    df["turnovers_per_36"] = df["turnovers_per_36"].fillna(df["turnovers"] / minutes * 36)
    attempts = df["field_goal_attempts"] + 0.44 * df["free_throw_attempts"]
    df["shot_attempts_per_36"] = df["shot_attempts_per_36"].fillna(attempts / minutes * 36)
    df["three_point_attempt_rate"] = df["three_point_attempt_rate"].fillna(df["three_point_attempts"] / df["field_goal_attempts"].replace(0, np.nan))
    df["free_throw_attempt_rate"] = df["free_throw_attempt_rate"].fillna(df["free_throw_attempts"] / df["field_goal_attempts"].replace(0, np.nan))
    df["usage_proxy"] = df["usage_proxy"].fillna((df["field_goal_attempts"] + 0.44 * df["free_throw_attempts"] + df["turnovers"]) / minutes)
    df["assist_to_turnover_ratio"] = df["assist_to_turnover_ratio"].fillna(df["assists"] / df["turnovers"].replace(0, np.nan))
    ts_den = 2 * (df["field_goal_attempts"] + 0.44 * df["free_throw_attempts"])
    df["ts_pct"] = df["ts_pct"].fillna(df["points"] / ts_den.replace(0, np.nan))
    if "source_id" not in df.columns:
        df["source_id"] = df.get("source", "unknown")
    return df


def split_years(df: pd.DataFrame) -> list[int]:
    pos_years = sorted(pd.to_numeric(df.loc[df["signed_cba_next_season"] == 1, "season_start_year"], errors="coerce").dropna().unique())
    years = []
    for year in pos_years:
        train = df[df["season_start_year"] < year]
        if train["signed_cba_next_season"].nunique() == 2:
            years.append(int(year))
    return years


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(skipna=True)
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return ((values - values.mean(skipna=True)) / std).fillna(0.0)


def metric(y: pd.Series, score: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    score = pd.to_numeric(score, errors="coerce").fillna(-999)
    ranked = pd.DataFrame({"y": y.to_numpy(), "score": score.to_numpy()}).sort_values("score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    pos = int(ranked["y"].sum())
    base = pos / n if n else 0.0
    out = {
        "candidate_count": n,
        "test_positive_count": pos,
        "base_positive_rate": base,
        "pr_auc": float(average_precision_score(y, score)) if pos else 0.0,
        "mrr": 0.0,
    }
    hit_idx = ranked.index[ranked["y"].eq(1)].tolist()
    if hit_idx:
        out["mrr"] = 1 / (hit_idx[0] + 1)
    for k in TOP_K:
        top = ranked.head(k)
        hits = int(top["y"].sum())
        prec = hits / len(top) if len(top) else 0.0
        rec = hits / pos if pos else 0.0
        cum_hits = top["y"].cumsum()
        out[f"precision_at_{k}"] = prec
        out[f"recall_at_{k}"] = rec
        out[f"lift_at_{k}"] = prec / base if base else 0.0
        out[f"hit_count_at_{k}"] = hits
        out[f"map_at_{k}"] = float(((cum_hits / np.arange(1, len(top) + 1)) * top["y"]).sum() / min(pos, k)) if pos else 0.0
    return out


def select_features(df: pd.DataFrame, extra_exclude: set[str] | None = None) -> tuple[list[str], list[str]]:
    exclude = LEAKAGE_COLUMNS | (extra_exclude or set())
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude or col == "season":
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and pd.to_numeric(df[col], errors="coerce").notna().any():
            numeric.append(col)
        elif col in {"league", "source_id", "role_cluster_label", "height_group", "nationality_region"} and df[col].notna().any():
            categorical.append(col)
    return numeric, categorical


def make_logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
                        (
                            "categorical",
                            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                            categorical,
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=SEED)),
        ]
    )


def rule_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [zscore(df[col]) if col in df.columns else pd.Series(0.0, index=df.index) for col in cols]
    return sum(parts) / max(len(parts), 1)


def high_usage_pool(df: pd.DataFrame) -> pd.DataFrame:
    med = df.groupby(["league", "season"], dropna=False)["usage_proxy"].transform("median")
    return df[(df["games"] >= 10) & (df["minutes_per_game"] >= 15) & (df["usage_proxy"] > med)].copy()


def time_aware_common_pool(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    leagues = set(train.loc[train["signed_cba_next_season"] == 1, "league"].dropna().astype(str))
    return train[train["league"].astype(str).isin(leagues)].copy(), test[test["league"].astype(str).isin(leagues)].copy()


def ranker_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
