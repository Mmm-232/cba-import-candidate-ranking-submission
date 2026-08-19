from __future__ import annotations

import argparse
import pandas as pd

try:
    from .utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs
except ImportError:
    from utils import PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


def run() -> None:
    ensure_data_dirs()
    df = pd.read_csv(PROCESSED_DIR / "labelled_candidate_dataset_source_aware.csv")
    pos = df.groupby("source_id", dropna=False).agg(rows=("player_name_key", "size"), positives=("signed_cba_next_season", "sum"), unique_players=("player_name_key", "nunique")).reset_index()
    pos["positive_rate"] = pos["positives"] / pos["rows"]
    pos.to_csv(REPORTS_DIR / "source_contribution_to_positives.csv", index=False)

    top_path = REPORTS_DIR / "source_aware_top_true_positives.csv"
    if top_path.exists():
        top = pd.read_csv(top_path)
        rows = []
        for k in [20, 50, 100]:
            subset = top[top["rank"] <= k] if "rank" in top.columns else pd.DataFrame()
            if not subset.empty:
                part = subset.groupby("source_id", dropna=False).size().reset_index(name=f"top_{k}_true_positive_hits")
                part["k"] = k
                rows.append(part)
        hits = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["source_id", "k", "top_k_true_positive_hits"])
    else:
        hits = pd.DataFrame(columns=["source_id", "k", "top_k_true_positive_hits"])
    hits.to_csv(REPORTS_DIR / "source_contribution_to_topk_hits.csv", index=False)

    by_season = df.groupby(["source_id", "season"], dropna=False).agg(rows=("player_name_key", "size"), positives=("signed_cba_next_season", "sum")).reset_index()
    by_season["positive_rate"] = by_season["positives"] / by_season["rows"]
    by_season.to_csv(REPORTS_DIR / "source_coverage_by_season.csv", index=False)
    print("Wrote source contribution reports")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
