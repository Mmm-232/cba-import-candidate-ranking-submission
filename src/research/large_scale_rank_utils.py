from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import player_season_rank_utils as ps
from .source_diagnostics_utils import add_source_group, safe_rank_percentile


TOP_K = [20, 50, 100, 200, 300]


COMMON_PATHWAY_TERMS = [
    "g league",
    "euroleague",
    "eurocup",
    "australian-nbl",
    "australian nbl",
    "israeli",
    "winner",
    "italian",
    "lega",
    "german",
    "bbl",
    "french",
    "lnb",
    "jeep",
    "adriatic",
    "aba",
]


def stable_id(*parts: object) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_ps(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")
    df = add_source_group(df)
    df["candidate_id"] = [stable_id(r.player_name_key, r.season) for r in df.itertuples(index=False)]
    return df


def is_common_pathway(row: pd.Series) -> bool:
    text = f"{row.get('league', '')} {row.get('leagues_played_that_season', '')} {row.get('source_group', '')}".lower()
    if any(term in text for term in COMMON_PATHWAY_TERMS):
        return True
    return bool(row.get("has_common_pathway_league", 0))


def high_confidence_pathway_mask(df: pd.DataFrame) -> pd.Series:
    rows = []
    for row in df.itertuples(index=False):
        prior = df[
            (df["season_start_year"] < row.season_start_year)
            & (df["signed_cba_next_season"].eq(1))
        ]
        leagues = set(prior["league"].dropna().astype(str))
        sources = set(prior["source_group"].dropna().astype(str))
        row_leagues = str(getattr(row, "leagues_played_that_season", "")).split(";")
        ok = str(getattr(row, "league", "")) in leagues or str(getattr(row, "source_group", "")) in sources
        ok = ok or any(l.strip() in leagues for l in row_leagues)
        rows.append(ok)
    return pd.Series(rows, index=df.index)


def pool_mask(df: pd.DataFrame, pool: str) -> pd.Series:
    if pool == "pool_full_multisource":
        return pd.Series(True, index=df.index)
    if pool == "pool_all_eligible_overseas":
        return pd.Series(True, index=df.index)
    if pool == "pool_common_cba_source_leagues":
        return df.apply(is_common_pathway, axis=1)
    if pool == "pool_high_confidence_pathways":
        return high_confidence_pathway_mask(df)
    raise ValueError(pool)


def metric(y: pd.Series, score: pd.Series) -> dict[str, float]:
    score = pd.to_numeric(score, errors="coerce").fillna(-999)
    ranked = pd.DataFrame({"y": y.astype(int), "score": score}).sort_values("score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    pos = int(ranked["y"].sum())
    base = pos / n if n else 0
    out = {"candidate_count": n, "test_positive_count": pos, "base_positive_rate": base, "mrr": 0.0}
    hits_idx = ranked.index[ranked["y"].eq(1)].tolist()
    if hits_idx:
        out["mrr"] = 1 / (hits_idx[0] + 1)
    for k in TOP_K:
        top = ranked.head(k)
        hits = int(top["y"].sum())
        precision = hits / len(top) if len(top) else 0.0
        recall = hits / pos if pos else 0.0
        denom = min(pos, k)
        if denom:
            cum_hits = top["y"].cumsum()
            ap = float(((cum_hits / np.arange(1, len(top) + 1)) * top["y"]).sum() / denom)
        else:
            ap = 0.0
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"hit_count_at_{k}"] = hits
        out[f"lift_at_{k}"] = precision / base if base else 0.0
        out[f"map_at_{k}"] = ap
    return out


def score_model(train: pd.DataFrame, test: pd.DataFrame, model: str, gleague_boost: float = 0.0) -> pd.Series | None:
    if test.empty:
        return None
    if model == "rule_based":
        score = ps.fit_rule_score(test)
    elif model == "cba_fit_score_baseline":
        score = pd.to_numeric(test.get("cba_import_fit_score"), errors="coerce").fillna(0)
    elif model == "logistic_regression_balanced":
        score = ps.logistic_scores(train, test)
    else:
        raise ValueError(model)
    if score is None:
        return None
    return pd.to_numeric(score, errors="coerce").fillna(-999) + gleague_boost * test["source_group"].eq("gleague_nba_api").astype(float)


def top_true_positives(test: pd.DataFrame, score: pd.Series, meta: dict) -> pd.DataFrame:
    ranked = test.copy()
    ranked["score"] = pd.to_numeric(score, errors="coerce").fillna(-999).to_numpy()
    ranked["rank"] = ranked["score"].rank(method="first", ascending=False).astype(int)
    for k, v in meta.items():
        ranked[k] = v
    return ranked[ranked["signed_cba_next_season"].eq(1)].sort_values("rank").head(30)
