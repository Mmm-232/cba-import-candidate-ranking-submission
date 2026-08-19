from __future__ import annotations

import pandas as pd

try:
    from .utils import nba_season_label, season_start_year
except ImportError:  # Allows direct script-style imports in some contexts.
    from utils import nba_season_label, season_start_year


def next_season_label(season: object) -> str | pd.NA:
    try:
        return f"{season_start_year(str(season)) + 1}-{season_start_year(str(season)) + 2}"
    except ValueError:
        return pd.NA


def add_cba_next_season_labels(candidates: pd.DataFrame, cba_labels: pd.DataFrame) -> pd.DataFrame:
    labelled = candidates.copy()
    labelled["next_season"] = labelled["season"].map(next_season_label)
    positive_pairs = set(zip(cba_labels["player_name_key"], cba_labels["cba_season"]))
    labelled["signed_cba_next_season"] = [
        int((row.player_name_key, row.next_season) in positive_pairs) for row in labelled.itertuples(index=False)
    ]
    return labelled
