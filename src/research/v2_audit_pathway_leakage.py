from __future__ import annotations

import pandas as pd

from .utils import is_chinese_cba_league, is_nba_league
from .v2_pathway_utils import V2_PROCESSED_DIR, V2_REPORTS_DIR, ensure_v2_dirs


DATA = V2_PROCESSED_DIR / "labelled_player_season_dataset_pathway_features.csv"
FEATURES = V2_REPORTS_DIR / "v2_model_feature_sets.csv"
RESULTS = V2_REPORTS_DIR / "v2_model_results.csv"
OUTPUT = V2_REPORTS_DIR / "v2_pathway_leakage_audit.csv"

TARGET_TERMS = [
    "signed_cba_next_season",
    "signed_cba_within_2_seasons",
    "signed_cba_within_3_seasons",
    "signed_cba_within_5_seasons",
    "ever_signed_cba_after_t",
    "first_cba_season_after_t",
    "seasons_until_cba",
]


def run() -> None:
    ensure_v2_dirs()
    df = pd.read_csv(DATA)
    rows = []
    rows.append(
        {
            "check": "chinese_cba_rows_excluded_from_overseas_input",
            "passed": int(not df["league"].map(is_chinese_cba_league).any()),
            "details": f"chinese_cba_rows={int(df['league'].map(is_chinese_cba_league).sum())}",
        }
    )
    rows.append(
        {
            "check": "nba_inclusion_explicit",
            "passed": int("v2_include_nba" in df.columns and not df["v2_include_nba"].fillna(False).astype(bool).any()),
            "details": f"nba_rows_present={int(df['league'].map(is_nba_league).sum())}; v2_include_nba_column={'v2_include_nba' in df.columns}",
        }
    )
    if FEATURES.exists():
        feat = pd.read_csv(FEATURES)
        feature_text = "; ".join(feat.get("features", pd.Series(dtype=str)).dropna().astype(str)).lower()
        bad = [term for term in TARGET_TERMS if term.lower() in feature_text]
        rows.append({"check": "target_columns_not_in_feature_sets", "passed": int(not bad), "details": "; ".join(bad) if bad else "no target columns found"})
    else:
        rows.append({"check": "target_columns_not_in_feature_sets", "passed": 0, "details": "feature set file missing"})

    future_feature_cols = [c for c in df.columns if ("after_t" in c or "within_" in c or "seasons_until_cba" in c or "first_cba" in c) and c not in TARGET_TERMS]
    rows.append(
        {
            "check": "future_cba_not_used_as_feature_columns",
            "passed": int(not future_feature_cols),
            "details": "; ".join(future_feature_cols) if future_feature_cols else "no unexpected future-looking feature columns",
        }
    )
    prior_like = [c for c in df.columns if c.endswith("_before_t") or "last_seen_gap" in c or "immediate_previous" in c]
    rows.append({"check": "career_history_features_marked_time_bounded", "passed": int(bool(prior_like)), "details": f"time-bounded feature count={len(prior_like)}"})

    if RESULTS.exists():
        res = pd.read_csv(RESULTS)
        rows.append(
            {
                "check": "walk_forward_split_outputs_exist",
                "passed": int("test_year" in res.columns and "train_rows" in res.columns and (pd.to_numeric(res["train_rows"], errors="coerce") > 0).any()),
                "details": f"result_rows={len(res)}",
            }
        )
    else:
        rows.append({"check": "walk_forward_split_outputs_exist", "passed": 0, "details": "model results missing"})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(f"Wrote leakage audit to {OUTPUT}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
