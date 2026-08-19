from __future__ import annotations

import argparse
import pandas as pd

try:
    from .utils import DATA_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs
except ImportError:
    from utils import DATA_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_data_dirs


def run() -> None:
    ensure_data_dirs()
    df = pd.read_csv(PROCESSED_DIR / "labelled_candidate_dataset_multisource_verified.csv")
    registry_path = DATA_DIR / "config" / "data_sources_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    rows = []
    for sid, group in df.groupby("source_id", dropna=False):
        reg = registry[registry["source_id"].astype(str) == str(sid)].head(1)
        rows.append(
            {
                "source_id": sid,
                "rows_contributed": len(group),
                "positives_contributed": int(group["signed_cba_next_season"].sum()),
                "seasons_covered": f"{group['season'].min()} to {group['season'].max()}",
                "leagues_covered": group["league"].nunique(),
                "source_type": reg["source_type"].iloc[0] if not reg.empty else "",
                "status": reg["status"].iloc[0] if not reg.empty else "",
                "used_for": reg["used_for"].iloc[0] if not reg.empty else "features",
                "has_evidence_url_or_source_note": bool(
                    ("source_url_or_file" in group.columns and group["source_url_or_file"].notna().any())
                    or ("source_note" in group.columns and group["source_note"].notna().any())
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "source_evidence_summary.csv", index=False)
    md = "# Source Evidence Summary\n\n"
    md += "| source_id | rows | positives | seasons | leagues | type | status | evidence/source note |\n"
    md += "|---|---:|---:|---|---:|---|---|---:|\n"
    for row in out.itertuples(index=False):
        md += f"| {row.source_id} | {row.rows_contributed} | {row.positives_contributed} | {row.seasons_covered} | {row.leagues_covered} | {row.source_type} | {row.status} | {row.has_evidence_url_or_source_note} |\n"
    (REPORTS_DIR / "source_evidence_summary.md").write_text(md, encoding="utf-8")
    print("Wrote source evidence summary")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
