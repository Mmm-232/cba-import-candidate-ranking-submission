from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ..utils import (
    PROCESSED_DIR,
    REPORTS_DIR,
    add_data_completeness_score,
    add_derived_history_metrics,
    ensure_data_dirs,
    is_chinese_cba_league,
    is_nba_league,
    player_name_key,
    safe_divide,
)


DEFAULT_OUTPUT = PROCESSED_DIR / "new_candidates_clean.csv"
DEFAULT_REPORT = REPORTS_DIR / "new_candidate_ingestion_report.csv"

REQUIRED_COLUMNS = ["player_name_raw", "season", "league", "team"]
CORE_STATS = [
    "games",
    "minutes",
    "minutes_per_game",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "ts_pct",
    "usage_proxy",
    "points_per_36",
]

OUTPUT_COLUMNS = [
    "player_name_raw",
    "player_name_key",
    "season",
    "league",
    "team",
    "source",
    *CORE_STATS,
    "age",
    "height",
    "weight",
    "position",
    "country",
    "official_player_url",
    "official_stats_url",
    "data_completeness_score",
    "missing_required_fields",
    "missing_core_stats_count",
    "data_quality_warning",
]

COLUMN_ALIASES = {
    "player": "player_name_raw",
    "name": "player_name_raw",
    "player_name": "player_name_raw",
    "playername": "player_name_raw",
    "year": "season",
    "season_year": "season",
    "club": "team",
    "squad": "team",
    "competition": "league",
    "league_name": "league",
    "mpg": "minutes_per_game",
    "pts": "points",
    "reb": "rebounds",
    "trb": "rebounds",
    "ast": "assists",
    "stl": "steals",
    "blk": "blocks",
    "tov": "turnovers",
    "to": "turnovers",
    "fga": "field_goal_attempts",
    "3pa": "three_point_attempts",
    "tpa": "three_point_attempts",
    "fta": "free_throw_attempts",
    "fg%": "fg_pct",
    "3p%": "three_pct",
    "ft%": "ft_pct",
}


# 功能：清理输入字段名，方便识别不同写法。
def _normalise_col(col: object) -> str:
    text = str(col).strip().lower()
    text = text.replace("%", "%")
    text = re.sub(r"[\s\-\/]+", "_", text)
    text = text.strip("_")
    return COLUMN_ALIASES.get(text, COLUMN_ALIASES.get(text.replace("_", ""), text))


# 功能：读取 CSV、Excel 或 JSON 候选人文件。
def _load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported input file type: {path.suffix}")


# 功能：把用户文件中的字段映射到项目统一字段。
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen: dict[str, int] = {}
    columns = []
    for col in df.columns:
        mapped = _normalise_col(col)
        seen[mapped] = seen.get(mapped, 0) + 1
        columns.append(mapped if seen[mapped] == 1 else f"{mapped}_{seen[mapped]}")
    df.columns = columns
    return df


# 功能：记录一条输入数据检查警告。
def _warning(row: pd.Series) -> str:
    warnings = []
    if row.get("missing_required_fields"):
        warnings.append("missing_required_fields")
    if int(row.get("missing_core_stats_count", 0) or 0) >= 10:
        warnings.append("low_performance_stat_coverage")
    if pd.to_numeric(pd.Series([row.get("data_completeness_score")]), errors="coerce").fillna(0).iloc[0] < 0.35:
        warnings.append("low_completeness")
    return "; ".join(warnings)


# 功能：读取并检查用户候选人文件，生成可评分的标准表。
def ingest_candidates(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    source_name: str | None = None,
    season: str | None = None,
    include_nba: bool = False,
    include_chinese_cba: bool = False,
) -> pd.DataFrame:
    ensure_data_dirs()
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    raw = _load_input(input_path)
    df = _normalise_columns(raw)
    input_rows = len(df)

    for col in REQUIRED_COLUMNS + CORE_STATS + ["source", "age", "height", "weight", "position", "country", "official_player_url", "official_stats_url"]:
        if col not in df.columns:
            df[col] = pd.NA
    if source_name:
        df["source"] = df["source"].fillna(source_name)
        df.loc[df["source"].astype(str).str.strip().eq(""), "source"] = source_name
    if season:
        df["season"] = df["season"].fillna(season)
        df.loc[df["season"].astype(str).str.strip().eq(""), "season"] = season

    for col in CORE_STATS + ["age", "weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["player_name_raw"] = df["player_name_raw"].astype(str).replace({"nan": ""}).str.strip()
    df["player_name_key"] = df["player_name_raw"].map(player_name_key)

    missing_required_mask = pd.Series(False, index=df.index)
    missing_fields = []
    for col in REQUIRED_COLUMNS:
        missing = df[col].isna() | df[col].astype(str).str.strip().isin(["", "nan", "None"])
        missing_required_mask = missing_required_mask | missing
        missing_fields.append(missing.map(lambda x, c=col: c if x else ""))
    df["missing_required_fields"] = pd.concat(missing_fields, axis=1).apply(lambda r: "; ".join([x for x in r if x]), axis=1)

    nba_mask = df["league"].map(is_nba_league)
    chinese_cba_mask = df["league"].map(is_chinese_cba_league)
    eligible_mask = ~missing_required_mask
    if not include_nba:
        eligible_mask = eligible_mask & ~nba_mask
    if not include_chinese_cba:
        eligible_mask = eligible_mask & ~chinese_cba_mask

    df = add_derived_history_metrics(df)
    games = pd.to_numeric(df["games"], errors="coerce")
    minutes = pd.to_numeric(df["minutes"], errors="coerce")
    points = pd.to_numeric(df["points"], errors="coerce")
    fga = pd.to_numeric(df["field_goal_attempts"], errors="coerce")
    fta = pd.to_numeric(df["free_throw_attempts"], errors="coerce")
    turnovers = pd.to_numeric(df["turnovers"], errors="coerce")
    df["minutes_per_game"] = pd.to_numeric(df["minutes_per_game"], errors="coerce").fillna(safe_divide(minutes, games))
    df["points_per_36"] = pd.to_numeric(df["points_per_36"], errors="coerce").fillna(safe_divide(points, minutes) * 36)
    df["usage_proxy"] = pd.to_numeric(df["usage_proxy"], errors="coerce").fillna(safe_divide(fga + 0.44 * fta + turnovers, minutes))
    ts_denominator = 2 * (fga + 0.44 * fta)
    df["ts_pct"] = pd.to_numeric(df["ts_pct"], errors="coerce").fillna(safe_divide(points, ts_denominator))
    df = add_data_completeness_score(df, CORE_STATS)
    df["missing_core_stats_count"] = df[CORE_STATS].isna().sum(axis=1)
    df["data_quality_warning"] = df.apply(_warning, axis=1)

    valid = df[eligible_mask].copy()
    for col in OUTPUT_COLUMNS:
        if col not in valid.columns:
            valid[col] = pd.NA
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid[OUTPUT_COLUMNS].to_csv(output_path, index=False)

    enough_perf = valid[["minutes_per_game", "points_per_36", "usage_proxy", "ts_pct"]].notna().any(axis=1)
    report = pd.DataFrame(
        [
            {"metric": "input_rows", "value": input_rows},
            {"metric": "valid_rows", "value": len(valid)},
            {"metric": "excluded_rows", "value": input_rows - len(valid)},
            {"metric": "excluded_nba_rows", "value": int((nba_mask & ~include_nba).sum())},
            {"metric": "excluded_chinese_cba_rows", "value": int((chinese_cba_mask & ~include_chinese_cba).sum())},
            {"metric": "rows_missing_required_fields", "value": int(missing_required_mask.sum())},
            {"metric": "rows_with_enough_performance_data", "value": int(enough_perf.sum())},
            {"metric": "rows_with_low_completeness", "value": int(pd.to_numeric(valid["data_completeness_score"], errors="coerce").fillna(0).lt(0.35).sum())},
            {"metric": "include_nba", "value": include_nba},
            {"metric": "include_chinese_cba", "value": include_chinese_cba},
            {"metric": "input_file", "value": str(input_path)},
            {"metric": "output_file", "value": str(output_path)},
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(f"Wrote cleaned new candidates to {output_path}")
    print(f"Wrote ingestion report to {report_path}")
    print(report.to_string(index=False))
    return valid[OUTPUT_COLUMNS]


# 功能：执行用户候选文件读取、校验和标准化流程。
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest user-provided new CBA import candidate player-season data.")
    parser.add_argument("--input", required=True, help="Path to CSV, Excel, or JSON file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Cleaned output CSV path.")
    parser.add_argument("--source-name", default=None, help="Optional source name to fill missing source values.")
    parser.add_argument("--season", default=None, help="Optional season to fill missing season values.")
    parser.add_argument("--include-nba", action="store_true", help="Include NBA rows. Default excludes NBA.")
    parser.add_argument("--include-chinese-cba", action="store_true", help="Include Chinese-CBA rows. Default excludes Chinese-CBA.")
    args = parser.parse_args()
    ingest_candidates(
        input_path=args.input,
        output_path=args.output,
        source_name=args.source_name,
        season=args.season,
        include_nba=args.include_nba,
        include_chinese_cba=args.include_chinese_cba,
    )


if __name__ == "__main__":
    main()
