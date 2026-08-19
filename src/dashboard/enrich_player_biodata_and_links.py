from __future__ import annotations

from datetime import date, datetime, timezone
import time
from pathlib import Path
import os
from urllib.parse import quote_plus

import pandas as pd

from ..utils import PROCESSED_DIR, REPORTS_DIR, player_name_key


FRONTEND_INPUT = REPORTS_DIR / "frontend_recommendations.csv"
PLAYER_SEASON_INPUT = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
MANUAL_OVERRIDES = PROCESSED_DIR.parent / "manual" / "player_biodata_overrides.csv"
OUTPUT = REPORTS_DIR / "frontend_recommendations_enriched.csv"
SUMMARY_OUTPUT = REPORTS_DIR / "biodata_enrichment_summary.csv"
NBA_CACHE = REPORTS_DIR / "nba_biodata_cache.csv"
MAX_NBA_API_CALLS = int(os.environ.get("MAX_NBA_BIODATA_API_CALLS", "20"))

NBA_BIODATA_FIELDS = [
    "player_name_key",
    "player_name_raw",
    "nba_player_id",
    "height",
    "weight",
    "birthdate",
    "position",
    "country",
    "school",
    "last_affiliation",
    "season_exp",
    "rosterstatus",
    "biodata_source",
    "biodata_match_confidence",
    "biodata_match_method",
    "cache_status",
    "cached_at",
]


def _load_nba_cache() -> pd.DataFrame:
    if NBA_CACHE.exists():
        return pd.read_csv(NBA_CACHE)
    return pd.DataFrame(columns=NBA_BIODATA_FIELDS)


def _save_nba_cache(cache: pd.DataFrame) -> None:
    NBA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    for col in NBA_BIODATA_FIELDS:
        if col not in cache.columns:
            cache[col] = pd.NA
    cache[NBA_BIODATA_FIELDS].to_csv(NBA_CACHE, index=False)


def _is_nba_or_gleague(row: pd.Series) -> bool:
    text = " ".join(str(row.get(c, "")) for c in ["source", "source_id", "league"]).lower()
    return "gleague" in text or "g league" in text or "nba" in text


def _age_at_season_start(birthdate: object, recommendation_season: object) -> object:
    if pd.isna(birthdate) or pd.isna(recommendation_season):
        return pd.NA
    try:
        birth = pd.to_datetime(birthdate, errors="coerce")
        if pd.isna(birth):
            return pd.NA
        start_year = int(str(recommendation_season).split("-")[0].strip())
        season_start = date(start_year, 10, 1)
        return int(season_start.year - birth.year - ((season_start.month, season_start.day) < (birth.month, birth.day)))
    except (TypeError, ValueError):
        return pd.NA


def _google_search_url(query: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(query)


def _official_links(row: pd.Series) -> dict[str, object]:
    name = str(row.get("player_name_raw", "")).strip()
    league = str(row.get("league", "")).strip()
    source_text = " ".join(str(row.get(c, "")) for c in ["source", "source_id", "league"]).lower()
    nba_id = row.get("nba_player_id")
    out = {
        "official_player_url": pd.NA,
        "official_stats_url": pd.NA,
        "official_search_url": _google_search_url(f'"{name}" "{league}" official basketball stats'),
        "nba_stats_url": pd.NA,
        "euroleague_profile_or_search_url": pd.NA,
        "league_official_search_url": _google_search_url(f'"{name}" "{league}" official basketball stats'),
        "basketball_reference_search_url": _google_search_url(f'site:basketball-reference.com "{name}" basketball'),
    }
    if pd.notna(nba_id) and str(nba_id).strip() and str(nba_id).strip().lower() != "nan":
        out["nba_stats_url"] = f"https://www.nba.com/stats/player/{int(float(nba_id))}"
        out["official_player_url"] = out["nba_stats_url"]
        out["official_stats_url"] = out["nba_stats_url"]
    elif "gleague" in source_text or "g league" in source_text or "nba" in source_text:
        out["official_search_url"] = _google_search_url(f'site:nba.com/stats "{name}" basketball')
        out["league_official_search_url"] = out["official_search_url"]

    if "euroleague" in source_text or "euroleague" in league.lower():
        out["euroleague_profile_or_search_url"] = _google_search_url(f'site:euroleaguebasketball.net/euroleague/players "{name}"')
        out["league_official_search_url"] = out["euroleague_profile_or_search_url"]
    elif "eurocup" in source_text or "eurocup" in league.lower():
        out["euroleague_profile_or_search_url"] = _google_search_url(f'site:euroleaguebasketball.net/eurocup/players "{name}"')
        out["league_official_search_url"] = out["euroleague_profile_or_search_url"]
    return out


def _nba_static_match_map() -> tuple[dict[str, list[dict]], str]:
    try:
        from nba_api.stats.static import players
    except Exception as exc:  # pragma: no cover - depends on environment package state
        return {}, f"nba_api import failed: {exc}"
    mapping: dict[str, list[dict]] = {}
    try:
        for player in players.get_players():
            key = player_name_key(player.get("full_name", ""))
            mapping.setdefault(key, []).append(player)
    except Exception as exc:  # pragma: no cover
        return {}, f"nba_api static players failed: {exc}"
    return mapping, "ok"


def _fetch_common_player_info(nba_player_id: int) -> dict[str, object]:
    from nba_api.stats.endpoints import commonplayerinfo

    info = commonplayerinfo.CommonPlayerInfo(player_id=nba_player_id, timeout=12)
    frame = info.common_player_info.get_data_frame()
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return {
        "nba_player_id": nba_player_id,
        "birthdate": row.get("BIRTHDATE"),
        "height": row.get("HEIGHT"),
        "weight": row.get("WEIGHT"),
        "position": row.get("POSITION"),
        "country": row.get("COUNTRY"),
        "school": row.get("SCHOOL"),
        "last_affiliation": row.get("LAST_AFFILIATION"),
        "season_exp": row.get("SEASON_EXP"),
        "rosterstatus": row.get("ROSTERSTATUS"),
    }


def _nba_biodata_for_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    logs: list[str] = []
    cache = _load_nba_cache()
    cache_keys = set(cache.get("player_name_key", pd.Series(dtype=str)).dropna().astype(str))
    match_map, status = _nba_static_match_map()
    logs.append(f"nba_static_players_status={status}")
    if not match_map:
        return cache, logs

    needed = df[df.apply(_is_nba_or_gleague, axis=1)].copy()
    commonplayerinfo_available = True
    commonplayerinfo_calls = 0
    for _, row in needed.iterrows():
        key = str(row.get("player_name_key", "")).strip()
        if not key or key in cache_keys:
            continue
        candidates = match_map.get(key, [])
        if len(candidates) != 1:
            cache_row = {
                "player_name_key": key,
                "player_name_raw": row.get("player_name_raw"),
                "biodata_source": "nba_api",
                "biodata_match_confidence": "none" if not candidates else "low",
                "biodata_match_method": "nba_static_players_name_key",
                "cache_status": "no_confident_unique_match",
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            cache = pd.concat([cache, pd.DataFrame([cache_row])], ignore_index=True)
            cache_keys.add(key)
            continue
        player = candidates[0]
        cache_row = {
            "player_name_key": key,
            "player_name_raw": row.get("player_name_raw"),
            "nba_player_id": player.get("id"),
            "biodata_source": "nba_api",
            "biodata_match_confidence": "high",
            "biodata_match_method": "exact_player_name_key_unique",
            "cache_status": "matched_static_only",
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        if commonplayerinfo_available and commonplayerinfo_calls < MAX_NBA_API_CALLS:
            try:
                time.sleep(0.7)
                commonplayerinfo_calls += 1
                cache_row.update(_fetch_common_player_info(int(player.get("id"))))
                cache_row["cache_status"] = "commonplayerinfo_success"
            except Exception as exc:  # API/network errors should not block dashboard links.
                commonplayerinfo_available = False
                cache_row["cache_status"] = f"commonplayerinfo_failed: {exc}"
                logs.append(f"{row.get('player_name_raw')} commonplayerinfo_failed={exc}")
        else:
            cache_row["cache_status"] = "matched_static_only_commonplayerinfo_skipped_after_limit_or_failure"
        cache = pd.concat([cache, pd.DataFrame([cache_row])], ignore_index=True)
        cache_keys.add(key)
        _save_nba_cache(cache)

    _save_nba_cache(cache)
    logs.append(f"commonplayerinfo_calls={commonplayerinfo_calls}")
    logs.append(f"max_nba_api_calls={MAX_NBA_API_CALLS}")
    return cache, logs


def _merge_manual_overrides(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if not MANUAL_OVERRIDES.exists():
        return df, 0
    manual = pd.read_csv(MANUAL_OVERRIDES)
    if "player_name_key" not in manual.columns:
        return df, 0
    manual = manual.rename(columns={"notes": "biodata_notes", "birth_date": "birthdate"})
    manual_cols = [
        "player_name_key",
        "height",
        "weight",
        "birthdate",
        "age",
        "position",
        "country",
        "official_player_url",
        "official_stats_url",
        "biodata_source",
        "biodata_notes",
    ]
    manual_cols = [c for c in manual_cols if c in manual.columns]
    manual = manual[manual_cols].drop_duplicates("player_name_key", keep="last")
    df = df.merge(manual, on="player_name_key", how="left", suffixes=("", "_manual"))
    applied = 0
    for base in [c for c in manual_cols if c != "player_name_key"]:
        manual_col = f"{base}_manual"
        if manual_col in df.columns:
            mask = df[manual_col].notna()
            applied += int(mask.sum())
            df.loc[mask, base] = df.loc[mask, manual_col]
            df = df.drop(columns=[manual_col])
    return df, applied


def enrich() -> pd.DataFrame:
    if not FRONTEND_INPUT.exists():
        raise FileNotFoundError(f"Missing {FRONTEND_INPUT}. Run python -m src.dashboard.export_frontend_recommendations first.")

    df = pd.read_csv(FRONTEND_INPUT)
    if "player_name_key" not in df.columns:
        df["player_name_key"] = df["player_name_raw"].map(player_name_key)

    # Attach any already-present identifiers/URLs from the player-season file without changing model data.
    if PLAYER_SEASON_INPUT.exists():
        ps = pd.read_csv(PLAYER_SEASON_INPUT)
        possible_cols = [
            "player_name_key",
            "season",
            "source_id",
            "league",
            "team",
            "nba_player_id",
            "PERSON_ID",
            "person_id",
            "player_id",
            "official_player_url",
            "official_stats_url",
            "source_url",
            "source_url_or_file",
        ]
        keep = [c for c in possible_cols if c in ps.columns]
        if keep and "player_name_key" in keep:
            ps = ps[keep].drop_duplicates("player_name_key")
            df = df.merge(ps, on="player_name_key", how="left", suffixes=("", "_playerseason"))

    for id_col in ["nba_player_id", "PERSON_ID", "person_id", "player_id"]:
        if id_col in df.columns:
            df["nba_player_id"] = df.get("nba_player_id", pd.Series(pd.NA, index=df.index)).fillna(df[id_col])

    cache, logs = _nba_biodata_for_candidates(df)
    bio_cols = [c for c in NBA_BIODATA_FIELDS if c in cache.columns and c != "player_name_raw"]
    if "player_name_key" in cache.columns:
        cache_latest = cache[bio_cols].drop_duplicates("player_name_key", keep="last")
        df = df.merge(cache_latest, on="player_name_key", how="left", suffixes=("", "_nba"))
        for col in ["height", "weight", "birthdate", "position", "country", "last_affiliation", "biodata_source", "biodata_match_confidence", "biodata_match_method"]:
            nba_col = f"{col}_nba"
            if nba_col in df.columns:
                df[col] = df.get(col, pd.Series(pd.NA, index=df.index)).fillna(df[nba_col])
                df = df.drop(columns=[nba_col])

    df, manual_values_applied = _merge_manual_overrides(df)

    if "age_at_recommendation_season" not in df.columns:
        df["age_at_recommendation_season"] = pd.NA
    birth_col = "birthdate" if "birthdate" in df.columns else "birth_date"
    if birth_col in df.columns:
        calculated_age = df.apply(lambda r: _age_at_season_start(r.get(birth_col), r.get("recommendation_season")), axis=1)
        df["age_at_recommendation_season"] = df["age_at_recommendation_season"].fillna(calculated_age)
    if "age" in df.columns:
        df["age_at_recommendation_season"] = df["age_at_recommendation_season"].fillna(df["age"])

    link_rows = df.apply(_official_links, axis=1, result_type="expand")
    for col in link_rows.columns:
        if col in df.columns:
            df[col] = df[col].fillna(link_rows[col])
        else:
            df[col] = link_rows[col]

    if "birthdate" not in df.columns and "birth_date" in df.columns:
        df["birthdate"] = df["birth_date"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    summary_rows = [
        {"metric": "input_rows", "value": len(df)},
        {"metric": "players_with_height", "value": int(df.get("height", pd.Series(dtype=object)).notna().sum())},
        {"metric": "players_with_weight", "value": int(df.get("weight", pd.Series(dtype=object)).notna().sum())},
        {"metric": "players_with_birthdate", "value": int(df.get("birthdate", pd.Series(dtype=object)).notna().sum())},
        {"metric": "players_with_age_at_recommendation_season", "value": int(df.get("age_at_recommendation_season", pd.Series(dtype=object)).notna().sum())},
        {"metric": "direct_nba_stats_links", "value": int(df.get("nba_stats_url", pd.Series(dtype=object)).notna().sum())},
        {"metric": "euroleague_or_eurocup_profile_or_search_links", "value": int(df.get("euroleague_profile_or_search_url", pd.Series(dtype=object)).notna().sum())},
        {"metric": "league_official_search_links", "value": int(df.get("league_official_search_url", pd.Series(dtype=object)).notna().sum())},
        {"metric": "manual_override_file_exists", "value": MANUAL_OVERRIDES.exists()},
        {"metric": "manual_override_values_applied", "value": manual_values_applied},
    ]
    for log in logs:
        summary_rows.append({"metric": "log", "value": log})
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUTPUT, index=False)
    print(f"Wrote enriched recommendations to {OUTPUT}")
    print(f"Wrote biodata enrichment summary to {SUMMARY_OUTPUT}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return df


if __name__ == "__main__":
    enrich()
