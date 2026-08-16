from __future__ import annotations

from io import StringIO

import pandas as pd

from .utils import DATA_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_data_dirs


CACHE_BEFORE = DATA_DIR / "cache" / "gleague_scaling_before_fix_mapped.csv"
STAT_COLS = [
    "games",
    "minutes",
    "minutes_per_game",
    "points",
    "field_goal_attempts",
    "free_throw_attempts",
    "turnovers",
    "points_per_36",
    "usage_proxy",
    "usage_rate",
    "ts_pct",
    "efg_pct",
    "data_completeness_score",
]


# 函数：_ensure_derived
def _ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["games", "minutes", "minutes_per_game", "points", "field_goal_attempts", "free_throw_attempts", "turnovers"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    minutes = out["minutes"].replace(0, pd.NA)
    if "points_per_36" not in out.columns:
        out["points_per_36"] = out["points"] / minutes * 36
    if "usage_proxy" not in out.columns:
        out["usage_proxy"] = (out["field_goal_attempts"] + 0.44 * out["free_throw_attempts"] + out["turnovers"]) / minutes
    return out


# 函数：_profile
def _profile(df: pd.DataFrame, label: str) -> dict[str, object]:
    row = {"dataset": label, "rows": len(df)}
    for col in STAT_COLS:
        s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="Float64")
        row[f"{col}_mean"] = s.mean()
        row[f"{col}_median"] = s.median()
        row[f"{col}_non_null_pct"] = s.notna().mean() if len(s) else 0
    mpg = pd.to_numeric(df["minutes_per_game"], errors="coerce") if "minutes_per_game" in df.columns else pd.Series(pd.NA, index=df.index)
    active = df[mpg.ge(10)].copy()
    row["active_rows"] = len(active)
    row["active_points_per_36_lt_1"] = int(pd.to_numeric(active.get("points_per_36"), errors="coerce").lt(1).sum())
    row["active_usage_proxy_lt_0_05"] = int(pd.to_numeric(active.get("usage_proxy"), errors="coerce").lt(0.05).sum())
    return row


# 函数：run
def run() -> None:
    ensure_data_dirs()
    (DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "gleague_nba_api_candidate_pool_raw.csv"
    mapped_path = PROCESSED_DIR / "gleague_nba_api_candidate_pool_mapped.csv"
    ps_path = PROCESSED_DIR / "labelled_player_season_dataset_gleague.csv"
    diag_path = REPORTS_DIR / "gleague_positive_rank_diagnostics.csv"

    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    mapped = _ensure_derived(pd.read_csv(mapped_path)) if mapped_path.exists() else pd.DataFrame()
    if not CACHE_BEFORE.exists() and not mapped.empty:
        mapped.to_csv(CACHE_BEFORE, index=False)
    before = _ensure_derived(pd.read_csv(CACHE_BEFORE)) if CACHE_BEFORE.exists() else pd.DataFrame()
    ps = pd.read_csv(ps_path) if ps_path.exists() else pd.DataFrame()
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()

    rows = []
    rows.append(_profile(before, "before_fix_snapshot"))
    rows.append(_profile(mapped, "current_mapped"))
    if not ps.empty:
        rows.append(_profile(ps[ps.get("has_gleague_row", pd.Series(False, index=ps.index)).eq(1)], "player_season_gleague_rows"))
    if not diag.empty:
        rows.append(_profile(diag, "gleague_positive_rank_diagnostics"))
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORTS_DIR / "gleague_stat_scaling_audit.csv", index=False)

    raw_modes = pd.DataFrame()
    if not raw.empty:
        raw_modes = raw.groupby(["api_season", "measure_type"], dropna=False).agg(
            rows=("PLAYER_ID", "count"),
            columns=("measure_type", lambda _: "; ".join(raw.columns)),
        ).reset_index()

    current = rows[1] if len(rows) > 1 else {}
    bug_likely = bool(
        current.get("minutes_per_game_median", 0) and current.get("minutes_per_game_median", 0) >= 10
        and current.get("points_per_36_median", 999) < 8
        and current.get("usage_proxy_median", 999) < 0.10
    )
    buffer = StringIO()
    audit.to_csv(buffer, index=False)
    raw_buffer = StringIO()
    raw_modes.to_csv(raw_buffer, index=False)
    lines = [
        "# G League Stat Scaling Audit",
        "",
        f"- PerGame vs totals scaling bug likely: {'yes' if bug_likely else 'no'}",
        "- Raw audit detects the current raw file by `api_season` and `measure_type`; earlier raw collection did not store explicit `per_mode`, but the API call used PerGame.",
        "- A before-fix snapshot is stored at `data/cache/gleague_scaling_before_fix_mapped.csv` if it did not already exist.",
        "",
        "## Raw Response Modes",
        "",
        "```csv",
        raw_buffer.getvalue().strip(),
        "```",
        "",
        "## Scaling Profiles",
        "",
        "```csv",
        buffer.getvalue().strip(),
        "```",
        "",
        "## Correction Required",
        "",
        "Use PerGame values consistently by multiplying per-game counting stats by GP before computing totals-based derived features, or use Base Totals if available. Do not divide per-game points/FGA/FTA/TOV by total minutes.",
    ]
    (REPORTS_DIR / "gleague_stat_scaling_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote G League stat scaling audit; bug_likely={bug_likely}")


if __name__ == "__main__":
    run()
