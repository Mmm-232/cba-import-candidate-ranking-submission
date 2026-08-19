from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguedashplayerstats
except ImportError:  # pragma: no cover - report gracefully at runtime
    leaguedashplayerstats = None

try:
    from .utils import CACHE_DIR, REPORTS_DIR, ensure_data_dirs
except ImportError:
    from utils import CACHE_DIR, REPORTS_DIR, ensure_data_dirs


SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
MEASURE_TYPES = ["Base", "Advanced"]
REQUIRED_FIELDS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GP",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGA",
    "FG3A",
    "FTA",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "USG_PCT",
    "TS_PCT",
    "EFG_PCT",
]


def _cache_paths(season: str, measure_type: str) -> tuple[Path, Path]:
    cache_dir = CACHE_DIR / "gleague_nba_api"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = f"leaguedashplayerstats_league20_{season}_{measure_type}".replace("-", "_").lower()
    return cache_dir / f"{slug}.csv", cache_dir / f"{slug}.json"


def _fetch(season: str, measure_type: str, delay_seconds: float = 1.5, retries: int = 1) -> pd.DataFrame:
    csv_path, json_path = _cache_paths(season, measure_type)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    if leaguedashplayerstats is None:
        raise RuntimeError("nba_api is not installed.")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(delay_seconds * 2)
        try:
            endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable="20",
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense=measure_type,
                timeout=45,
            )
            df = endpoint.get_data_frames()[0]
            df.to_csv(csv_path, index=False)
            json_path.write_text(endpoint.get_json(), encoding="utf-8")
            time.sleep(delay_seconds)
            return df
        except Exception as exc:  # noqa: BLE001 - endpoint errors are reported, not hidden
            last_error = exc
    raise RuntimeError(str(last_error))


def _audit_one(season: str, measure_type: str) -> dict[str, object]:
    row: dict[str, object] = {
        "season": season,
        "league_id": "20",
        "measure_type": measure_type,
        "success": False,
        "row_count": 0,
        "returned_columns": "",
        "sample_player_names": "",
        "error_message": "",
        "extraction_date": date.today().isoformat(),
    }
    try:
        df = _fetch(season, measure_type)
        row["success"] = True
        row["row_count"] = len(df)
        row["returned_columns"] = "; ".join(map(str, df.columns))
        if "PLAYER_NAME" in df.columns:
            row["sample_player_names"] = "; ".join(df["PLAYER_NAME"].dropna().astype(str).head(5))
    except Exception as exc:  # noqa: BLE001
        row["error_message"] = str(exc)
        df = pd.DataFrame()

    columns = set(df.columns)
    for field in REQUIRED_FIELDS:
        row[f"has_{field.lower()}"] = field in columns
    return row


def _write_markdown(audit: pd.DataFrame) -> None:
    ok = audit[audit["success"].eq(True) & audit["row_count"].gt(0)]
    base = audit[audit["measure_type"].eq("Base")]
    advanced = audit[audit["measure_type"].eq("Advanced")]
    worked = sorted(ok["season"].unique())
    failed = sorted(audit.loc[~audit["success"].eq(True) | audit["row_count"].eq(0), "season"].unique())
    base_ok = base["success"].eq(True).any() and base["row_count"].gt(0).any()
    adv_ok = advanced["success"].eq(True).any() and advanced["row_count"].gt(0).any()
    key_basic = ["has_player_id", "has_player_name", "has_team_abbreviation", "has_gp", "has_min", "has_pts", "has_reb", "has_ast", "has_fga", "has_fg3a", "has_fta"]
    key_adv = ["has_usg_pct", "has_ts_pct", "has_efg_pct"]
    sufficient_basic = bool(base_ok and base[key_basic].any(axis=None))
    sufficient_adv = bool(adv_ok and advanced[key_adv].any(axis=None))
    usable = bool(base_ok and not ok.empty)

    lines = [
        "# G League NBA API Feasibility",
        "",
        f"- Tested LeagueID: `20` only",
        f"- Tested seasons: {', '.join(SEASONS)}",
        f"- Successful seasons with rows: {', '.join(worked) if worked else 'None'}",
        f"- Failed or empty seasons: {', '.join(failed) if failed else 'None'}",
        f"- Base stats available: {'yes' if base_ok else 'no'}",
        f"- Advanced stats available: {'yes' if adv_ok else 'no'}",
        f"- Basic schema fields sufficient: {'yes' if sufficient_basic else 'no'}",
        f"- Advanced fields sufficient: {'yes' if sufficient_adv else 'partial/no'}",
        f"- Usable enough to build a G League candidate pool: {'yes' if usable else 'no'}",
        f"- Local CSV fallback still needed: {'yes' if not usable or not adv_ok else 'yes, as a reproducible backup'}",
        "",
        "Notes: no NBA LeagueID `00` requests are made by this audit.",
    ]
    (REPORTS_DIR / "gleague_nba_api_feasibility.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> pd.DataFrame:
    ensure_data_dirs()
    rows = []
    for season in SEASONS:
        for measure_type in MEASURE_TYPES:
            rows.append(_audit_one(season, measure_type))
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORTS_DIR / "gleague_nba_api_feasibility.csv", index=False)
    _write_markdown(audit)
    print(f"Wrote G League NBA API feasibility audit: {len(audit)} checks")
    return audit


def main() -> None:
    run()


if __name__ == "__main__":
    main()
