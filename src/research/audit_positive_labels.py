from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

try:
    from .build_labels import next_season_label
    from .utils import DATA_DIR, PROCESSED_DIR, REPORTS_DIR, configure_logging, ensure_data_dirs, normalise_player_name, player_name_key
except ImportError:
    from build_labels import next_season_label
    from utils import DATA_DIR, PROCESSED_DIR, REPORTS_DIR, configure_logging, ensure_data_dirs, normalise_player_name, player_name_key


LOGGER = logging.getLogger(__name__)
KAGGLE_DIR = DATA_DIR / "external" / "kaggle_49leagues"


def _load_kaggle_chinese_cba() -> pd.DataFrame:
    frames = []
    for path in sorted(KAGGLE_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        df["_source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    league = raw.get("League", pd.Series("", index=raw.index)).astype(str).str.lower()
    cba = raw[league.str.contains("chinese|china|cba", na=False)].copy()
    if cba.empty:
        return cba
    cba["cba_season"] = cba["Season"].astype(str).str.replace(" ", "", regex=False)
    cba["player_name_raw"] = cba["Player"]
    cba["player_name_clean"] = cba["player_name_raw"].map(normalise_player_name)
    cba["player_name_key"] = cba["player_name_clean"].map(player_name_key)
    return cba


def _candidate_key(df: pd.DataFrame) -> pd.Series:
    return df["player_name_key"].astype(str) + "|" + df["next_season"].astype(str)


def audit_positive_labels(labelled_path: Path, enriched_path: Path, cba_path: Path) -> None:
    labelled = pd.read_csv(labelled_path)
    enriched = pd.read_csv(enriched_path) if enriched_path.exists() else labelled.copy()
    cba = pd.read_csv(cba_path)
    cba["cba_key"] = cba["player_name_key"].astype(str) + "|" + cba["cba_season"].astype(str)

    before = int(labelled["signed_cba_next_season"].sum())
    cba_pairs = set(cba["cba_key"])

    checked = labelled.copy()
    checked["expected_next_cba_season"] = checked["season"].map(next_season_label)
    checked["pair_key"] = checked["player_name_key"].astype(str) + "|" + checked["expected_next_cba_season"].astype(str)
    should_be_positive = checked["pair_key"].isin(cba_pairs)
    high_confidence_mask = should_be_positive & checked["signed_cba_next_season"].eq(0)

    possible_rows = []
    for row in checked[high_confidence_mask].itertuples(index=False):
        match = cba[cba["cba_key"] == row.pair_key].head(1)
        possible_rows.append(
            {
                "player_name_raw": row.player_name_raw,
                "player_name_key": row.player_name_key,
                "overseas_season": row.season,
                "expected_next_cba_season": row.expected_next_cba_season,
                "matched_cba_name_if_any": match["player_name_raw"].iloc[0] if not match.empty else "",
                "reason_suspected_missed": "exact_player_key_and_next_season_match_not_labelled",
                "confidence": "high",
            }
        )

    # Fuzzy/manual review candidates: overseas candidates in season t whose next-season CBA names are very similar.
    cba_by_season = {season: group for season, group in cba.groupby("cba_season")}
    for season, group in checked[checked["signed_cba_next_season"].eq(0)].groupby("expected_next_cba_season"):
        cba_group = cba_by_season.get(season)
        if cba_group is None or cba_group.empty:
            continue
        choices = cba_group["player_name_clean"].dropna().tolist()
        for row in group.itertuples(index=False):
            match = process.extractOne(row.player_name_clean, choices, scorer=fuzz.token_sort_ratio) if choices else None
            if match and match[1] >= 90:
                matched_name = match[0]
                matched_key = cba_group.loc[cba_group["player_name_clean"] == matched_name, "player_name_key"].iloc[0]
                if matched_key != row.player_name_key:
                    possible_rows.append(
                        {
                            "player_name_raw": row.player_name_raw,
                            "player_name_key": row.player_name_key,
                            "overseas_season": row.season,
                            "expected_next_cba_season": row.expected_next_cba_season,
                            "matched_cba_name_if_any": matched_name,
                            "reason_suspected_missed": "high_similarity_name_different_key_manual_review",
                            "confidence": "medium",
                        }
                    )

    possible = pd.DataFrame(possible_rows).drop_duplicates() if possible_rows else pd.DataFrame(
        columns=[
            "player_name_raw",
            "player_name_key",
            "overseas_season",
            "expected_next_cba_season",
            "matched_cba_name_if_any",
            "reason_suspected_missed",
            "confidence",
        ]
    )

    raw_cba = _load_kaggle_chinese_cba()
    name_issues = []
    if not raw_cba.empty:
        existing_keys = set(cba["player_name_key"].dropna().astype(str))
        missing_cba_players = raw_cba[~raw_cba["player_name_key"].astype(str).isin(existing_keys)].copy()
        for row in missing_cba_players.drop_duplicates(["player_name_key", "cba_season"]).itertuples(index=False):
            name_issues.append(
                {
                    "player_name_raw": row.player_name_raw,
                    "player_name_key": row.player_name_key,
                    "season": row.cba_season,
                    "issue_type": "chinese_cba_player_not_in_cba_imports_extended",
                    "confidence": "manual_review",
                }
            )
    name_issues_df = pd.DataFrame(name_issues) if name_issues else pd.DataFrame(
        columns=["player_name_raw", "player_name_key", "season", "issue_type", "confidence"]
    )

    corrected = labelled.copy()
    recovered = 0
    if high_confidence_mask.any():
        corrected.loc[high_confidence_mask, "signed_cba_next_season"] = 1
        recovered = int(high_confidence_mask.sum())

    after = int(corrected["signed_cba_next_season"].sum())
    summary = pd.DataFrame(
        [
            {"metric": "positives_before_audit", "value": before},
            {"metric": "high_confidence_positives_recovered", "value": recovered},
            {"metric": "positives_after_audit", "value": after},
            {
                "metric": "uncertain_possible_positives_needing_manual_review",
                "value": int((possible["confidence"] != "high").sum()) if not possible.empty else 0,
            },
            {"metric": "name_matching_issues", "value": len(name_issues_df)},
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "positive_label_audit_summary.csv", index=False)
    possible.to_csv(REPORTS_DIR / "possible_missed_positive_labels.csv", index=False)
    name_issues_df.to_csv(REPORTS_DIR / "name_matching_issues.csv", index=False)
    corrected.to_csv(PROCESSED_DIR / "labelled_candidate_dataset_label_audited.csv", index=False)
    if enriched_path.exists():
        enriched_corrected = enriched.copy()
        key_cols = ["player_name_key", "season", "league", "team"]
        label_map = corrected[key_cols + ["signed_cba_next_season"]].drop_duplicates(key_cols)
        enriched_corrected = enriched_corrected.drop(columns=["signed_cba_next_season"], errors="ignore").merge(
            label_map,
            on=key_cols,
            how="left",
        )
        enriched_corrected["signed_cba_next_season"] = enriched_corrected["signed_cba_next_season"].fillna(0).astype(int)
        enriched_corrected.to_csv(PROCESSED_DIR / "labelled_candidate_dataset_enriched_label_audited.csv", index=False)

    LOGGER.info("Positive audit complete: before=%s recovered=%s after=%s", before, recovered, after)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit possible missed positive CBA labels.")
    parser.add_argument("--labelled", type=Path, default=PROCESSED_DIR / "labelled_candidate_dataset.csv")
    parser.add_argument("--enriched", type=Path, default=PROCESSED_DIR / "labelled_candidate_dataset_enriched.csv")
    parser.add_argument("--cba", type=Path, default=PROCESSED_DIR / "cba_imports_extended.csv")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    ensure_data_dirs()
    audit_positive_labels(args.labelled, args.enriched, args.cba)


if __name__ == "__main__":
    main()
