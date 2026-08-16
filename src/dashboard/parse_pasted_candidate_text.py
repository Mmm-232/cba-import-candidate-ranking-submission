from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

import pandas as pd


UPLOAD_OUTPUT = Path("data/manual/uploads/pasted_candidate_text_latest.csv")
REPORT_OUTPUT = Path("data/reports/pasted_candidate_parse_report.csv")

FORMAT_MAP = {
    "auto": "auto",
    "basketball_reference_per36": "basketball_reference_per36",
    "basketball_reference_per_game": "basketball_reference_per_game",
    "basketball_reference_advanced": "basketball_reference_advanced",
    "generic_csv_text": "generic_csv_text",
    "Auto detect": "auto",
    "Basketball-Reference Per 36": "basketball_reference_per36",
    "Basketball-Reference Per Game": "basketball_reference_per_game",
    "Basketball-Reference Advanced": "basketball_reference_advanced",
    "Generic CSV text": "generic_csv_text",
}

CORE = {
    "Player",
    "Tm",
    "Age",
    "G",
    "GS",
    "MP",
    "PTS",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGA",
    "3PA",
    "FTA",
    "FG%",
    "3P%",
    "FT%",
}
ADVANCED = {
    "PER",
    "TS%",
    "eFG%",
    "3PAr",
    "FTr",
    "ORB%",
    "DRB%",
    "TRB%",
    "AST%",
    "STL%",
    "BLK%",
    "TOV%",
    "USG%",
    "OWS",
    "DWS",
    "WS",
    "WS/48",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",
}
CONTEXT = {"Pos", "Team", "League", "Season", "Source"}
IGNORED = {"Rk", "Awards", "", "Unnamed: 0"}

BREF_BASE_MAP = {
    "Player": "player_name_raw",
    "Tm": "team",
    "Team": "team",
    "Age": "age",
    "Pos": "position",
    "G": "games",
    "GS": "games_started",
    "MP": "minutes",
    "FG%": "fg_pct",
    "3P%": "three_pct",
    "FT%": "ft_pct",
    "Season": "season",
    "League": "league",
    "Source": "source",
}
PER36_MAP = {
    "PTS": "points_per_36",
    "TRB": "rebounds_per_36",
    "AST": "assists_per_36",
    "STL": "steals_per_36",
    "BLK": "blocks_per_36",
    "TOV": "turnovers_per_36",
    "FGA": "field_goal_attempts_per_36",
    "3PA": "three_point_attempts_per_36",
    "FTA": "free_throw_attempts_per_36",
}
PER_GAME_MAP = {
    "PTS": "points",
    "TRB": "rebounds",
    "AST": "assists",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "FGA": "field_goal_attempts",
    "3PA": "three_point_attempts",
    "FTA": "free_throw_attempts",
}
ADVANCED_MAP = {
    "PER": "per",
    "TS%": "ts_pct",
    "eFG%": "efg_pct",
    "3PAr": "three_point_attempt_rate",
    "FTr": "free_throw_rate",
    "ORB%": "offensive_rebound_rate",
    "DRB%": "defensive_rebound_rate",
    "TRB%": "total_rebound_rate",
    "AST%": "assist_rate",
    "STL%": "steal_rate",
    "BLK%": "block_rate",
    "TOV%": "turnover_rate",
    "USG%": "usage_rate",
    "OWS": "offensive_win_shares",
    "DWS": "defensive_win_shares",
    "WS": "win_shares",
    "WS/48": "win_shares_per_48",
    "OBPM": "obpm",
    "DBPM": "dbpm",
    "BPM": "bpm",
    "VORP": "vorp",
}


# 功能：清理粘贴文本中的空行和无关字符。
def _clean_lines(text: str) -> list[str]:
    lines = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("share & export") or lower.startswith("glossary") or lower.startswith("provided by"):
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


# 功能：判断粘贴表格使用逗号、制表符还是其他分隔符。
def _sniff_sep(lines: list[str]) -> str:
    first = lines[0] if lines else ""
    if "," in first:
        return ","
    if "\t" in first:
        return "\t"
    return r"\s+"


# 功能：把粘贴文本读取成表格。
def _read_table(text: str) -> pd.DataFrame:
    lines = _clean_lines(text)
    if not lines:
        return pd.DataFrame()
    sep = _sniff_sep(lines)
    cleaned = "\n".join(lines)
    if sep in {",", "\t"}:
        df = pd.read_csv(io.StringIO(cleaned), sep=sep)
    else:
        df = pd.read_csv(io.StringIO(cleaned), sep=sep, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    if "Player" in df.columns:
        df = df[df["Player"].astype(str).ne("Player")]
    return df


# 功能：判断粘贴数据属于 Per 36、Per Game 还是普通表格。
def _detect_format(df: pd.DataFrame, raw_text: str, hint: str) -> str:
    hint = FORMAT_MAP.get(hint, hint or "auto")
    if hint != "auto":
        return hint
    cols = set(df.columns.astype(str))
    lower_text = raw_text.lower()
    if {"PER", "TS%", "USG%", "WS", "BPM", "VORP"} & cols:
        return "basketball_reference_advanced"
    if "per 36" in lower_text or "per36" in lower_text:
        return "basketball_reference_per36"
    if {"Player", "Tm", "G", "MP", "PTS", "TRB", "AST"}.issubset(cols):
        pts = pd.to_numeric(df.get("PTS"), errors="coerce")
        mp = pd.to_numeric(df.get("MP"), errors="coerce")
        if mp.notna().any() and mp.median() > 60 and {"FGA", "3PA", "FTA"}.issubset(cols):
            return "basketball_reference_per36"
        if pts.notna().any() and mp.notna().any() and pts.median() > 8 and mp.median() < 40:
            return "basketball_reference_per36"
        return "basketball_reference_per_game"
    if {"Player", "Tm", "G", "MP"} & cols and ({"PTS", "PER", "TS%"} & cols):
        return "basketball_reference_per_game"
    return "generic_csv_text"


# 功能：识别并映射粘贴表格中的统计字段。
def _classify_columns(columns: list[str]) -> dict[str, str]:
    out = {}
    for col in columns:
        if col in IGNORED or col.startswith("Unnamed"):
            out[col] = "ignored"
        elif col in CORE:
            out[col] = "useful_core"
        elif col in ADVANCED:
            out[col] = "useful_advanced"
        elif col in CONTEXT:
            out[col] = "useful_context"
        elif col in {"player_name_raw", "team", "league", "season", "source"}:
            out[col] = "useful_display"
        else:
            out[col] = "unknown"
    return out


# 功能：把表格列安全转换成数值。
def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


# 功能：删除同一球员赛季的重复输入记录。
def _deduplicate(df: pd.DataFrame, keep_all_team_rows: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    if keep_all_team_rows or "player_name_raw" not in df.columns:
        return df, {"duplicate_players_detected": 0, "tot_rows_kept": 0, "team_rows_dropped": 0}
    duplicate_mask = df["player_name_raw"].duplicated(keep=False)
    duplicate_players = df.loc[duplicate_mask, "player_name_raw"].nunique()
    keep_indices = []
    dropped = 0
    tot_kept = 0
    for _, group in df.groupby("player_name_raw", dropna=False):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        team = group.get("team", pd.Series("", index=group.index)).astype(str)
        tot = group[team.str.upper().eq("TOT")]
        if not tot.empty:
            keep_indices.append(tot.index[0])
            dropped += len(group) - 1
            tot_kept += 1
        elif "minutes" in group.columns:
            mp = _safe_numeric(group["minutes"])
            keep_indices.append(mp.idxmax() if mp.notna().any() else group.index[0])
            dropped += len(group) - 1
        else:
            keep_indices.append(group.index[0])
            dropped += len(group) - 1
    return df.loc[keep_indices].copy(), {
        "duplicate_players_detected": int(duplicate_players),
        "tot_rows_kept": int(tot_kept),
        "team_rows_dropped": int(dropped),
    }


# 功能：解析粘贴的球员统计文本并生成统一候选表。
def parse_pasted_text(
    raw_text: str,
    season: str | None = None,
    league: str | None = None,
    source_name: str | None = "pasted_stats_table",
    format_hint: str = "auto",
    keep_all_team_rows: bool = False,
    output_path: str | Path = UPLOAD_OUTPUT,
    report_path: str | Path = REPORT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_lines = _clean_lines(raw_text)
    raw = _read_table(raw_text)
    detected = _detect_format(raw, raw_text, format_hint)
    classification = _classify_columns(raw.columns.astype(str).tolist())
    useful_core = [c for c, v in classification.items() if v == "useful_core"]
    useful_adv = [c for c, v in classification.items() if v == "useful_advanced"]
    ignored = [c for c, v in classification.items() if v == "ignored"]
    unknown = [c for c, v in classification.items() if v == "unknown"]

    out = pd.DataFrame(index=raw.index)
    for src, dest in BREF_BASE_MAP.items():
        if src in raw.columns:
            out[dest] = raw[src]
    if detected == "basketball_reference_per36":
        for src, dest in PER36_MAP.items():
            if src in raw.columns:
                out[dest] = _safe_numeric(raw[src])
        if {"PTS", "FGA", "FTA"}.issubset(raw.columns):
            pts, fga, fta = _safe_numeric(raw["PTS"]), _safe_numeric(raw["FGA"]), _safe_numeric(raw["FTA"])
            out["ts_pct"] = pts / (2 * (fga + 0.44 * fta)).replace(0, pd.NA)
        if {"FGA", "FTA", "TOV"}.issubset(raw.columns):
            fga, fta, tov = _safe_numeric(raw["FGA"]), _safe_numeric(raw["FTA"]), _safe_numeric(raw["TOV"])
            out["usage_proxy"] = (fga + 0.44 * fta + tov) / 36
    elif detected == "basketball_reference_per_game":
        for src, dest in PER_GAME_MAP.items():
            if src in raw.columns:
                out[dest] = _safe_numeric(raw[src])
    elif detected == "basketball_reference_advanced":
        for src, dest in ADVANCED_MAP.items():
            if src in raw.columns:
                out[dest] = _safe_numeric(raw[src])
        if "USG%" in raw.columns:
            out["usage_proxy_from_advanced"] = _safe_numeric(raw["USG%"])
    else:
        # Generic text: retain common already-standard columns where possible.
        for col in raw.columns:
            norm = str(col).strip().lower()
            if norm in {"player_name_raw", "season", "league", "team", "source"}:
                out[norm] = raw[col]

    if season:
        out["season"] = season
    if league:
        out["league"] = league
    if source_name:
        out["source"] = source_name
    missing_required = [c for c in ["player_name_raw", "season", "league", "team"] if c not in out.columns or out[c].isna().all()]
    if missing_required:
        out["missing_required_fields"] = "; ".join(missing_required)
    out, dup_report = _deduplicate(out, keep_all_team_rows)

    output_path = Path(output_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    warnings = []
    if "season" in missing_required:
        warnings.append("season missing; provide season override")
    if "league" in missing_required:
        warnings.append("league missing; provide league override")
    report = pd.DataFrame(
        [
            {"metric": "detected_format", "value": detected},
            {"metric": "input_line_count", "value": len(input_lines)},
            {"metric": "parsed_rows", "value": len(out)},
            {"metric": "parsed_columns", "value": len(raw.columns)},
            {"metric": "useful_core_columns", "value": "; ".join(useful_core)},
            {"metric": "useful_advanced_columns", "value": "; ".join(useful_adv)},
            {"metric": "ignored_columns", "value": "; ".join(ignored)},
            {"metric": "unknown_columns", "value": "; ".join(unknown)},
            {"metric": "missing_required_fields", "value": "; ".join(missing_required)},
            {"metric": "season_applied", "value": season or ""},
            {"metric": "league_applied", "value": league or ""},
            {"metric": "warnings", "value": "; ".join(warnings)},
            *[{"metric": k, "value": v} for k, v in dup_report.items()],
            {"metric": "output_path", "value": str(output_path)},
        ]
    )
    report.to_csv(report_path, index=False)
    print(f"Wrote parsed pasted text to {output_path}")
    print(f"Wrote parse report to {report_path}")
    print(report.to_string(index=False))
    return out, report


# 功能：执行粘贴表格解析并保存标准候选文件。
def main() -> None:
    parser = argparse.ArgumentParser(description="Parse pasted Basketball-Reference/stat table text into new-candidate CSV.")
    parser.add_argument("--input", required=True, help="Text file containing copied table text.")
    parser.add_argument("--season", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--source-name", default="pasted_stats_table")
    parser.add_argument("--format-hint", default="auto", choices=["auto", "basketball_reference_per36", "basketball_reference_per_game", "basketball_reference_advanced", "generic_csv_text"])
    parser.add_argument("--keep-all-team-rows", action="store_true")
    args = parser.parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    parse_pasted_text(text, args.season, args.league, args.source_name, args.format_hint, args.keep_all_team_rows)


if __name__ == "__main__":
    main()
