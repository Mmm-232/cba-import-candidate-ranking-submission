from __future__ import annotations

import pandas as pd

from .v2_pathway_utils import V2_REPORTS_DIR, add_future_cba_labels, ensure_v2_dirs, load_base_dataset, load_cba_labels, is_australian_nbl_row


def _top_players(group: pd.DataFrame) -> str:
    linked = group[group["ever_signed_cba_after_t"].eq(1)]
    names = linked.sort_values(["seasons_until_cba", "player_name_raw"], na_position="last")["player_name_raw"].dropna().astype(str).drop_duplicates().head(15)
    return "; ".join(names)


def run() -> None:
    ensure_v2_dirs()
    df = add_future_cba_labels(load_base_dataset(), load_cba_labels())
    df["league_for_audit"] = df["league"].fillna("Unknown").astype(str)
    rows = []
    for league, group in df.groupby("league_for_audit", dropna=False):
        gaps = pd.to_numeric(group.loc[group["ever_signed_cba_after_t"].eq(1), "seasons_until_cba"], errors="coerce").dropna()
        rows.append(
            {
                "league": league,
                "total_player_season_rows": len(group),
                "unique_players": group["player_name_key"].nunique(),
                "next_season_cba_positives": int(group["signed_cba_next_season"].sum()),
                "next_season_positive_rate": float(group["signed_cba_next_season"].mean()) if len(group) else 0.0,
                "within_2_seasons_count": int(group["signed_cba_within_2_seasons"].sum()),
                "within_3_seasons_count": int(group["signed_cba_within_3_seasons"].sum()),
                "within_5_seasons_count": int(group["signed_cba_within_5_seasons"].sum()),
                "any_future_cba_count": int(group["ever_signed_cba_after_t"].sum()),
                "average_gap_to_first_later_cba": float(gaps.mean()) if not gaps.empty else pd.NA,
                "median_gap_to_first_later_cba": float(gaps.median()) if not gaps.empty else pd.NA,
                "top_cba_linked_players": _top_players(group),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["any_future_cba_count", "within_3_seasons_count", "total_player_season_rows"], ascending=False)
    audit.to_csv(V2_REPORTS_DIR / "league_pathway_audit.csv", index=False)

    aus = df[df.apply(is_australian_nbl_row, axis=1)].copy()
    aus.to_csv(V2_REPORTS_DIR / "australian_nbl_pathway_audit.csv", index=False)

    positive_history = df[df["ever_signed_cba_after_t"].eq(1)].sort_values(["league_for_audit", "player_name_raw", "season_start_year"])
    positive_history[
        [
            "player_name_raw",
            "player_name_key",
            "season",
            "league_for_audit",
            "signed_cba_next_season",
            "signed_cba_within_2_seasons",
            "signed_cba_within_3_seasons",
            "signed_cba_within_5_seasons",
            "first_cba_season_after_t",
            "seasons_until_cba",
        ]
    ].to_csv(V2_REPORTS_DIR / "cba_positive_history_by_league.csv", index=False)

    name_issues = (
        df.groupby(["league_for_audit", "player_name_key"])["player_name_raw"]
        .nunique()
        .reset_index(name="raw_name_variants")
    )
    name_issues = name_issues[name_issues["raw_name_variants"].gt(1)]
    name_issues.to_csv(V2_REPORTS_DIR / "name_matching_issues_by_league.csv", index=False)
    print(f"Wrote league pathway audit to {V2_REPORTS_DIR / 'league_pathway_audit.csv'}")
    print(f"Australian-NBL rows audited: {len(aus)}")


if __name__ == "__main__":
    run()
