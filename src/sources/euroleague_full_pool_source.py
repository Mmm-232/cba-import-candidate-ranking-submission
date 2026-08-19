from __future__ import annotations

from datetime import date

import pandas as pd

from .euroleague_source import EuroleagueSource, COMPETITIONS, EUROLEAGUE_COMPLETENESS_COLUMNS

try:
    from ..utils import add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key
except ImportError:
    from utils import add_data_completeness_score, add_derived_history_metrics, normalise_player_name, player_name_key


# 类：EuroleagueFullPoolSource
# 类：EuroleagueFullPoolSource
class EuroleagueFullPoolSource(EuroleagueSource):
    name = "euroleague_api"
    source_name = "euroleague-api / EuroLeague and EuroCup API"

    def collect_full_pool(self, start_year: int = 2020, end_year: int = 2024) -> pd.DataFrame:
        frames = []
        for year in range(start_year, end_year + 1):
            for competition_code in COMPETITIONS:
                season_code = f"{competition_code}{year}"
                frame = self._fetch_competition_season(competition_code, season_code, year)
                if frame.empty:
                    continue
                frame["season"] = f"{year}-{year + 1}"
                frame["source_id"] = self.name
                frame["source_name"] = self.source_name
                frame["source_url_or_file"] = f"euroleague_api:{season_code}"
                frame["extraction_date"] = date.today().isoformat()
                frame["source_confidence"] = "api_player_stats"
                frame["verification_status"] = "source_extracted"
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["player_name_raw"] = out["source_player_name"]
        out["player_name_clean"] = out["player_name_raw"].map(normalise_player_name)
        out["player_name_key"] = out["player_name_clean"].map(player_name_key)
        out["next_season"] = out["season_start_year"].map(lambda y: f"{int(y)+1}-{int(y)+2}")
        out = add_derived_history_metrics(out)
        out = add_data_completeness_score(out, EUROLEAGUE_COMPLETENESS_COLUMNS)
        return out
