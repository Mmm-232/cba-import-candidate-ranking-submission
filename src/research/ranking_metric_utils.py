from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


TOP_K = [20, 50, 100, 300]


def ranking_metrics(y: pd.Series, score: pd.Series) -> dict[str, float]:
    y = y.astype(int).reset_index(drop=True)
    score = pd.to_numeric(score, errors="coerce").fillna(-999).reset_index(drop=True)
    ranked = pd.DataFrame({"y": y, "score": score}).sort_values("score", ascending=False).reset_index(drop=True)
    pos = int(ranked["y"].sum())
    n = len(ranked)
    base = pos / n if n else 0.0
    hit_idx = ranked.index[ranked["y"].eq(1)].tolist()
    out = {
        "candidate_count": n,
        "test_positive_count": pos,
        "base_positive_rate": base,
        "pr_auc": float(average_precision_score(y, score)) if pos else 0.0,
        "mrr": 1 / (hit_idx[0] + 1) if hit_idx else 0.0,
        "rank_of_first_true_positive": hit_idx[0] + 1 if hit_idx else pd.NA,
    }
    for k in TOP_K:
        top = ranked.head(k)
        hits = int(top["y"].sum())
        precision = hits / len(top) if len(top) else 0.0
        recall = hits / pos if pos else 0.0
        gains = top["y"].to_numpy()
        discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float((gains * discounts).sum())
        ideal = np.sort(ranked["y"].to_numpy())[::-1][:k]
        ideal_discounts = 1 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = float((ideal * ideal_discounts).sum())
        out[f"precision_at_{k}"] = precision
        out[f"recall_at_{k}"] = recall
        out[f"lift_at_{k}"] = precision / base if base else 0.0
        out[f"hit_count_at_{k}"] = hits
        out[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
    return out


def ranked_predictions(df: pd.DataFrame, score: pd.Series, meta: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    out["score"] = pd.to_numeric(score, errors="coerce").fillna(-999).to_numpy()
    out["rank"] = out["score"].rank(method="first", ascending=False).astype(int)
    for key, value in meta.items():
        out[key] = value
    return out
