from __future__ import annotations

import pandas as pd

from .v2_pathway_utils import PATHWAY_TERMS, V2_PROCESSED_DIR, V2_REPORTS_DIR, ensure_v2_dirs, league_text, load_cba_labels


INPUT = V2_PROCESSED_DIR / "labelled_player_season_dataset_pathway_labels.csv"
OUTPUT = V2_PROCESSED_DIR / "labelled_player_season_dataset_pathway_features.csv"
SUMMARY = V2_REPORTS_DIR / "pathway_feature_summary.csv"


def _contains_any(text: pd.Series, terms: list[str]) -> pd.Series:
    mask = pd.Series(False, index=text.index)
    for term in terms:
        mask = mask | text.str.contains(term, case=False, regex=False, na=False)
    return mask.astype(int)


def run() -> None:
    ensure_v2_dirs()
    df = pd.read_csv(INPUT)
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")
    df = df.sort_values(["player_name_key", "season_start_year", "season"]).copy()
    df["_league_text"] = df.apply(league_text, axis=1)

    for pathway, terms in PATHWAY_TERMS.items():
        df[f"is_{pathway}_row"] = _contains_any(df["_league_text"], terms)
        df[f"has_{pathway}_experience_before_t"] = (
            df.groupby("player_name_key")[f"is_{pathway}_row"].cumsum().groupby(df["player_name_key"]).shift(1).fillna(0).gt(0).astype(int)
        )

    df["number_of_prior_overseas_seasons"] = df.groupby("player_name_key").cumcount()
    prior_leagues = []
    last_leagues = []
    last_teams = []
    for _, group in df.groupby("player_name_key", sort=False):
        seen_leagues: set[str] = set()
        previous_league = pd.NA
        previous_team = pd.NA
        for idx, row in group.iterrows():
            prior_leagues.append((idx, len(seen_leagues)))
            last_leagues.append((idx, previous_league))
            last_teams.append((idx, previous_team))
            if pd.notna(row.get("league")):
                seen_leagues.add(str(row.get("league")))
            previous_league = row.get("league", pd.NA)
            previous_team = row.get("best_row_team", row.get("team", pd.NA))
    df["number_of_prior_overseas_leagues"] = pd.Series(dict(prior_leagues))
    df["last_overseas_league"] = pd.Series(dict(last_leagues))
    df["last_overseas_team"] = pd.Series(dict(last_teams))

    labels = load_cba_labels()
    cba_years = labels.groupby("player_name_key")["cba_start_year"].apply(lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))).to_dict()
    df["has_prior_cba_experience_before_t"] = [
        int(any(y < row.season_start_year for y in cba_years.get(row.player_name_key, [])))
        for row in df.itertuples(index=False)
    ]

    aus_last_year = {}
    aus_last_points = {}
    aus_last_usage = {}
    aus_last_ts = {}
    aus_seasons_count = {}
    for _, group in df.groupby("player_name_key", sort=False):
        last_year = pd.NA
        last_points = pd.NA
        last_usage = pd.NA
        last_ts = pd.NA
        aus_seen_seasons: set[str] = set()
        for idx, row in group.iterrows():
            aus_last_year[idx] = last_year
            aus_last_points[idx] = last_points
            aus_last_usage[idx] = last_usage
            aus_last_ts[idx] = last_ts
            aus_seasons_count[idx] = len(aus_seen_seasons)
            if int(row["is_australian_nbl_row"]) == 1:
                last_year = row["season_start_year"]
                last_points = row.get("points_per_36", pd.NA)
                last_usage = row.get("usage_proxy", pd.NA)
                last_ts = row.get("ts_pct", pd.NA)
                if pd.notna(row.get("season")):
                    aus_seen_seasons.add(str(row.get("season")))

    df["australian_nbl_seasons_before_t"] = pd.Series(aus_seasons_count)
    df["australian_nbl_last_seen_year"] = pd.Series(aus_last_year)
    df["australian_nbl_last_seen_gap"] = df["season_start_year"] - pd.to_numeric(df["australian_nbl_last_seen_year"], errors="coerce")
    df["australian_nbl_last_points_per_36"] = pd.Series(aus_last_points)
    df["australian_nbl_last_usage_proxy"] = pd.Series(aus_last_usage)
    df["australian_nbl_last_ts_pct"] = pd.Series(aus_last_ts)
    df["australian_nbl_is_immediate_previous_league"] = pd.to_numeric(df["australian_nbl_last_seen_gap"], errors="coerce").eq(1).astype(int)
    df["australian_nbl_any_experience_before_t"] = df["australian_nbl_seasons_before_t"].gt(0).astype(int)
    df["australian_nbl_current_or_prior_experience_to_t"] = (
        df["australian_nbl_any_experience_before_t"].eq(1) | df["is_australian_nbl_row"].eq(1)
    ).astype(int)

    df = df.drop(columns=["_league_text"], errors="ignore")
    df.to_csv(OUTPUT, index=False)

    feature_cols = [
        c
        for c in df.columns
        if c.startswith("has_")
        or c.startswith("australian_nbl_")
        or c.startswith("number_of_prior")
        or c in {"last_overseas_league", "last_overseas_team"}
    ]
    summary = []
    for col in feature_cols:
        series = df[col]
        summary.append(
            {
                "feature": col,
                "non_null": int(series.notna().sum()),
                "mean": float(pd.to_numeric(series, errors="coerce").mean()) if pd.to_numeric(series, errors="coerce").notna().any() else pd.NA,
            }
        )
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    print(f"Wrote pathway feature dataset to {OUTPUT}")
    print(f"Wrote pathway feature summary to {SUMMARY}")


if __name__ == "__main__":
    run()
