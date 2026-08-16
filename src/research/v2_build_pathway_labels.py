from __future__ import annotations

from .v2_pathway_utils import V2_PROCESSED_DIR, add_future_cba_labels, eligible_overseas_mask, ensure_v2_dirs, load_base_dataset, load_cba_labels


OUTPUT = V2_PROCESSED_DIR / "labelled_player_season_dataset_pathway_labels.csv"


# 函数：run
def run() -> None:
    ensure_v2_dirs()
    df = load_base_dataset()
    df["v2_include_nba"] = False
    df["v2_include_chinese_cba"] = False
    df["v2_eligible_overseas_input"] = eligible_overseas_mask(df, include_nba=False).astype(int)
    df = df[df["v2_eligible_overseas_input"].eq(1)].copy()
    labelled = add_future_cba_labels(df, load_cba_labels())
    labelled.to_csv(OUTPUT, index=False)
    print(f"Wrote pathway-labelled dataset to {OUTPUT}")
    print(f"Rows: {len(labelled)}")
    print(f"Next-season positives: {int(labelled['signed_cba_next_season'].sum())}")
    print(f"Within-2 positives: {int(labelled['signed_cba_within_2_seasons'].sum())}")
    print(f"Within-3 positives: {int(labelled['signed_cba_within_3_seasons'].sum())}")
    print(f"Within-5 positives: {int(labelled['signed_cba_within_5_seasons'].sum())}")


if __name__ == "__main__":
    run()
