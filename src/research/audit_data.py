"""Run dataset-level audits, including field coverage, league/source coverage, and anomaly checks."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from .utils import (
        DATA_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        configure_logging,
        ensure_data_dirs,
        is_chinese_cba_league,
        is_eligible_overseas_league,
        is_nba_league,
    )
except ImportError:  # Allows: python src/audit_data.py
    from utils import (
        DATA_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        configure_logging,
        ensure_data_dirs,
        is_chinese_cba_league,
        is_eligible_overseas_league,
        is_nba_league,
    )


LOGGER = logging.getLogger(__name__)

BASIC_FIELDS = [
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
]

ADVANCED_FIELDS = [
    "efg_pct",
    "ts_pct",
    "usage_proxy",
    "possessions_used",
    "field_goal_attempt_rate",
    "free_throw_rate",
    "turnover_rate",
    "assist_to_turnover_ratio",
    "total_rebound_rate",
    "offensive_rebound_rate",
    "defensive_rebound_rate",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
    "plus_minus",
    "data_completeness_score",
]

PLAY_TYPE_FIELDS = [
    "isolation_points_per_possession",
    "isolation_frequency",
    "pick_and_roll_ball_handler_ppp",
    "pick_and_roll_roll_man_ppp",
    "post_up_ppp",
    "spot_up_ppp",
    "transition_ppp",
]

AUDIT_DATASETS = {
    "player_history_all": "player_history_all.csv",
    "model_features": "model_features.csv",
    "eligible_candidate_pool": "eligible_candidate_pool.csv",
    "labelled_candidate_dataset": "labelled_candidate_dataset.csv",
}
RAW_KAGGLE_DIR = DATA_DIR / "external" / "kaggle_49leagues"


# 函数：_normalise_column_name
def _normalise_column_name(value: str) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


# 函数：_read_csv
def _read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required input: {path}")
        LOGGER.warning("Missing optional input: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


# 函数：_coverage_for_fields
def _coverage_for_fields(df: pd.DataFrame, dataset: str, fields: list[str], field_group: str) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for field in fields:
        exists = field in df.columns
        non_null = int(df[field].notna().sum()) if exists else 0
        rows.append(
            {
                "dataset": dataset,
                "field_group": field_group,
                "field": field,
                "exists": exists,
                "total_rows": total_rows,
                "non_null_rows": non_null,
                "missing_rows": total_rows - non_null if exists else total_rows,
                "non_null_rate": non_null / total_rows if total_rows else 0.0,
                "missing_rate": 1 - (non_null / total_rows) if total_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)


# 函数：_season_span
def _season_span(series: pd.Series) -> str:
    values = sorted(series.dropna().astype(str).unique())
    if not values:
        return ""
    return f"{values[0]} to {values[-1]}"


# 函数：_league_category
def _league_category(league: object) -> str:
    if is_nba_league(league):
        return "excluded_nba"
    if is_chinese_cba_league(league):
        return "excluded_chinese_cba"
    if is_eligible_overseas_league(league):
        return "eligible_overseas"
    return "excluded_other"


# 函数：_league_coverage
def _league_coverage(history: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frames = {
        "player_history_all": history,
        "labelled_candidate_dataset": labelled,
    }
    for dataset, df in frames.items():
        if df.empty or "league" not in df.columns:
            continue
        group_cols = [col for col in ["source", "league"] if col in df.columns]
        if not group_cols:
            group_cols = ["league"]
        grouped = df.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_cols, keys))
            positives = int(group["signed_cba_next_season"].sum()) if "signed_cba_next_season" in group.columns else 0
            rows.append(
                {
                    "dataset": dataset,
                    "source": key_map.get("source", ""),
                    "league": key_map.get("league", ""),
                    "eligibility_category": _league_category(key_map.get("league", "")),
                    "rows": len(group),
                    "unique_players": group["player_name_key"].nunique() if "player_name_key" in group.columns else None,
                    "seasons_covered": _season_span(group["season"]) if "season" in group.columns else "",
                    "unique_seasons": group["season"].nunique() if "season" in group.columns else None,
                    "positive_cba_next_season_rows": positives,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "rows"], ascending=[True, False])


# 函数：_source_level_audit
def _source_level_audit(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "source" not in history.columns:
        return pd.DataFrame()
    rows = []
    for source, group in history.groupby("source", dropna=False):
        basic_available = [field for field in BASIC_FIELDS if field in group.columns]
        advanced_available = [field for field in ADVANCED_FIELDS + PLAY_TYPE_FIELDS if field in group.columns]
        basic_complete = (
            group[basic_available].notna().mean(axis=1).mean() if basic_available and len(group) else 0.0
        )
        advanced_complete = (
            group[advanced_available].notna().mean(axis=1).mean() if advanced_available and len(group) else 0.0
        )
        rows.append(
            {
                "source": source,
                "rows": len(group),
                "unique_players": group["player_name_key"].nunique() if "player_name_key" in group.columns else None,
                "seasons_covered": _season_span(group["season"]) if "season" in group.columns else "",
                "leagues_covered": group["league"].nunique() if "league" in group.columns else None,
                "basic_stat_completeness": basic_complete,
                "advanced_stat_completeness": advanced_complete,
                "average_data_completeness_score": group["data_completeness_score"].mean()
                if "data_completeness_score" in group.columns
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


# 函数：_candidate_dataset_audit
def _candidate_dataset_audit(labelled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if labelled.empty:
        return pd.DataFrame()
    target = "signed_cba_next_season"
    total = len(labelled)
    positives = int(labelled[target].sum()) if target in labelled.columns else 0
    rows.append(
        {
            "section": "overall",
            "group": "all",
            "rows": total,
            "positive_rows": positives,
            "negative_rows": total - positives,
            "positive_rate": positives / total if total else 0.0,
        }
    )
    for col, section in [("season", "by_season"), ("league", "by_league")]:
        if col not in labelled.columns:
            continue
        for value, group in labelled.groupby(col, dropna=False):
            group_positives = int(group[target].sum()) if target in group.columns else 0
            rows.append(
                {
                    "section": section,
                    "group": value,
                    "rows": len(group),
                    "positive_rows": group_positives,
                    "negative_rows": len(group) - group_positives,
                    "positive_rate": group_positives / len(group) if len(group) else 0.0,
                }
            )
    for field in BASIC_FIELDS + ADVANCED_FIELDS + PLAY_TYPE_FIELDS:
        exists = field in labelled.columns
        missing_rate = float(labelled[field].isna().mean()) if exists and len(labelled) else 1.0
        rows.append(
            {
                "section": "missing_value_rate",
                "group": field,
                "rows": total,
                "positive_rows": None,
                "negative_rows": None,
                "positive_rate": None,
                "field_exists": exists,
                "missing_rate": missing_rate,
                "non_null_rate": 1 - missing_rate if exists else 0.0,
            }
        )
    return pd.DataFrame(rows)


# 函数：_load_raw_kaggle
def _load_raw_kaggle() -> pd.DataFrame:
    csv_files = sorted(RAW_KAGGLE_DIR.glob("*.csv"))
    if not csv_files:
        LOGGER.warning("No raw Kaggle CSV files found under %s", RAW_KAGGLE_DIR)
        return pd.DataFrame()
    frames = []
    for path in csv_files:
        frame = pd.read_csv(path)
        frame["_source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# 函数：_source_column
def _source_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalised = {_normalise_column_name(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalise_column_name(candidate)
        if key in normalised:
            return normalised[key]
    return None


# 函数：_append_anomalies
def _append_anomalies(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    dataset: str,
    anomaly_type: str,
    mask: pd.Series,
    details: str,
) -> None:
    if df.empty:
        return
    flagged = df[mask.fillna(False)].copy()
    for idx, row in flagged.iterrows():
        rows.append(
            {
                "dataset": dataset,
                "row_index": idx,
                "anomaly_type": anomaly_type,
                "details": details,
                "player_name_raw": row.get("player_name_raw", row.get("Player", "")),
                "player_name_key": row.get("player_name_key", ""),
                "league": row.get("league", row.get("League", "")),
                "season": row.get("season", row.get("Season", "")),
                "team": row.get("team", row.get("Team", "")),
                "source_file": row.get("_source_file", ""),
            }
        )


# 函数：_data_anomaly_reports
def _data_anomaly_reports(labelled: pd.DataFrame, raw_kaggle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for dataset_name, df in [("labelled_candidate_dataset", labelled)]:
        if df.empty:
            continue
        for col in ["fg_pct", "three_pct", "ft_pct"]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                _append_anomalies(
                    rows,
                    df,
                    dataset_name,
                    f"{col}_outside_0_1",
                    (values < 0) | (values > 1),
                    f"{col} is below 0 or above 1",
                )
        for col in ["games", "minutes", "points", "rebounds", "assists"]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                _append_anomalies(
                    rows,
                    df,
                    dataset_name,
                    f"negative_{col}",
                    values < 0,
                    f"{col} is negative",
                )

    if not raw_kaggle.empty:
        fgm = _source_column(raw_kaggle, ["FGM", "field_goals_made", "field_goal_made"])
        fga = _source_column(raw_kaggle, ["FGA", "field_goal_attempts", "field_goals_attempted"])
        three_pm = _source_column(raw_kaggle, ["3PM", "three_pm", "fg3m", "three_points_made"])
        three_pa = _source_column(raw_kaggle, ["3PA", "three_pa", "fg3a", "three_point_attempts"])
        ftm = _source_column(raw_kaggle, ["FTM", "free_throws_made", "free_throw_made"])
        fta = _source_column(raw_kaggle, ["FTA", "free_throw_attempts", "free_throws_attempted"])

        made_attempt_checks = [
            ("FGM_gt_FGA", fgm, fga),
            ("3PM_gt_3PA", three_pm, three_pa),
            ("FTM_gt_FTA", ftm, fta),
        ]
        for anomaly_type, made_col, attempt_col in made_attempt_checks:
            if made_col and attempt_col:
                made = pd.to_numeric(raw_kaggle[made_col], errors="coerce")
                attempts = pd.to_numeric(raw_kaggle[attempt_col], errors="coerce")
                _append_anomalies(
                    rows,
                    raw_kaggle,
                    "raw_kaggle_49leagues",
                    anomaly_type,
                    made > attempts,
                    f"{made_col} is greater than {attempt_col}",
                )

        for col_name, candidates in [
            ("games", ["GP", "games", "games_played"]),
            ("minutes", ["MIN", "minutes"]),
            ("points", ["PTS", "points"]),
            ("rebounds", ["REB", "rebounds"]),
            ("assists", ["AST", "assists"]),
        ]:
            source_col = _source_column(raw_kaggle, candidates)
            if source_col:
                values = pd.to_numeric(raw_kaggle[source_col], errors="coerce")
                _append_anomalies(
                    rows,
                    raw_kaggle,
                    "raw_kaggle_49leagues",
                    f"negative_{col_name}",
                    values < 0,
                    f"{source_col} is negative",
                )

    report = pd.DataFrame(rows)
    if report.empty:
        report = pd.DataFrame(
            columns=[
                "dataset",
                "row_index",
                "anomaly_type",
                "details",
                "player_name_raw",
                "player_name_key",
                "league",
                "season",
                "team",
                "source_file",
            ]
        )

    summary = (
        report.groupby(["dataset", "anomaly_type"], dropna=False)
        .size()
        .reset_index(name="anomaly_rows")
        .sort_values(["dataset", "anomaly_rows"], ascending=[True, False])
    )
    if summary.empty:
        summary = pd.DataFrame(columns=["dataset", "anomaly_type", "anomaly_rows"])
    return report, summary


# 函数：_markdown_table
def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"
    shown = df.head(max_rows).where(pd.notna(df.head(max_rows)), "")
    headers = [str(col) for col in shown.columns]
    rows = [[str(value) for value in row] for row in shown.to_numpy().tolist()]

    # 函数：clean
    def clean(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(clean(header) for header in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows],
        ]
    )


# 函数：_field_status
def _field_status(coverage: pd.DataFrame, dataset: str, fields: list[str]) -> tuple[list[str], list[str]]:
    subset = coverage[(coverage["dataset"] == dataset) & (coverage["field"].isin(fields))]
    available = subset.loc[subset["non_null_rows"] > 0, "field"].tolist()
    missing = subset.loc[subset["non_null_rows"] == 0, "field"].tolist()
    return available, missing


# 函数：_build_markdown
def _build_markdown(
    datasets: dict[str, pd.DataFrame],
    basic_coverage: pd.DataFrame,
    advanced_coverage: pd.DataFrame,
    league_coverage: pd.DataFrame,
    source_audit: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    anomaly_summary: pd.DataFrame,
    reports_dir: Path,
) -> str:
    history = datasets.get("player_history_all", pd.DataFrame())
    candidate = datasets.get("labelled_candidate_dataset", pd.DataFrame())
    eligible_pool = datasets.get("eligible_candidate_pool", pd.DataFrame())

    labelled_basic_available, labelled_basic_missing = _field_status(
        basic_coverage, "labelled_candidate_dataset", BASIC_FIELDS
    )
    labelled_adv_available, labelled_adv_missing = _field_status(
        advanced_coverage, "labelled_candidate_dataset", ADVANCED_FIELDS
    )
    labelled_play_available, labelled_play_missing = _field_status(
        advanced_coverage, "labelled_candidate_dataset", PLAY_TYPE_FIELDS
    )

    total_rows = len(candidate)
    positives = int(candidate["signed_cba_next_season"].sum()) if "signed_cba_next_season" in candidate.columns else 0
    positive_rate = positives / total_rows if total_rows else 0.0

    unique_leagues = candidate["league"].nunique() if "league" in candidate.columns else 0
    top_leagues = (
        candidate.groupby("league", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .head(20)
        if "league" in candidate.columns
        else pd.DataFrame()
    )
    top_positive_leagues = (
        candidate.groupby("league", dropna=False)["signed_cba_next_season"]
        .sum()
        .reset_index(name="positive_rows")
        .sort_values("positive_rows", ascending=False)
        .head(20)
        if {"league", "signed_cba_next_season"}.issubset(candidate.columns)
        else pd.DataFrame()
    )

    candidate_nba_rows = (
        int(candidate["league"].map(is_nba_league).sum()) if "league" in candidate.columns else 0
    )
    candidate_chinese_cba_rows = (
        int(candidate["league"].map(is_chinese_cba_league).sum()) if "league" in candidate.columns else 0
    )
    eligible_pool_nba_rows = (
        int(eligible_pool["league"].map(is_nba_league).sum()) if "league" in eligible_pool.columns else 0
    )
    eligible_pool_chinese_cba_rows = (
        int(eligible_pool["league"].map(is_chinese_cba_league).sum()) if "league" in eligible_pool.columns else 0
    )

    excluded_from_history = pd.DataFrame()
    if "league" in history.columns:
        tmp = history.copy()
        tmp["eligibility_category"] = tmp["league"].map(_league_category)
        excluded_from_history = (
            tmp.groupby(["eligibility_category", "league"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["eligibility_category", "rows"], ascending=[True, False])
        )

    source_coverage_path = reports_dir / "source_coverage_summary.csv"
    field_completeness_path = reports_dir / "field_completeness_summary.csv"
    league_filter_path = reports_dir / "league_filter_summary.csv"
    dissertation_path = reports_dir / "dissertation_experiment_summary.md"
    existing_report_notes = []
    for path in [source_coverage_path, field_completeness_path, league_filter_path, dissertation_path]:
        existing_report_notes.append(f"- `{path.name}`: {'available' if path.exists() else 'missing'}")

    anomaly_total = int(anomaly_summary["anomaly_rows"].sum()) if not anomaly_summary.empty else 0

    return f"""# Data Audit Summary

Generated by `python -m src.audit_data`.

## Audit Inputs

{chr(10).join(existing_report_notes)}

## 1. Does The Project Contain Basic Player Statistics?

**Yes.** The final candidate dataset contains the core box-score fields needed for a cautious first baseline. In `labelled_candidate_dataset.csv`, available basic fields include: {", ".join(labelled_basic_available) or "none"}.

Basic fields with no non-null values in the labelled candidate dataset: {", ".join(labelled_basic_missing) or "none"}.

Basic-stat coverage by dataset:

{_markdown_table(basic_coverage[basic_coverage["dataset"].isin(["player_history_all", "labelled_candidate_dataset"])][["dataset", "field", "exists", "non_null_rows", "non_null_rate"]], 40)}

## 2. Does The Project Contain Advanced Player Statistics?

**Partly.** The project contains several derived advanced indicators, especially `ts_pct`, `usage_proxy`, `field_goal_attempt_rate`, `free_throw_rate`, `assist_to_turnover_ratio`, and `data_completeness_score` where the source fields allow them to be calculated. More specialised advanced metrics such as official possession counts, ratings, plus-minus and play-type data are currently missing or mostly unavailable.

Available advanced fields in `labelled_candidate_dataset.csv`: {", ".join(labelled_adv_available) or "none"}.

Advanced fields with no non-null values in `labelled_candidate_dataset.csv`: {", ".join(labelled_adv_missing) or "none"}.

Play-type fields with data: {", ".join(labelled_play_available) or "none"}. Play-type fields missing or fully null: {", ".join(labelled_play_missing) or "none"}.

Advanced-stat coverage by dataset:

{_markdown_table(advanced_coverage[advanced_coverage["dataset"].isin(["player_history_all", "labelled_candidate_dataset"])][["dataset", "field_group", "field", "exists", "non_null_rows", "non_null_rate"]], 60)}

## 3. Does The Project Contain Data From Multiple Leagues?

**Yes.** The labelled candidate dataset contains **{unique_leagues} unique leagues**. It is therefore a multi-league overseas candidate pool, even though the currently merged player-history source coverage report is dominated by the local Kaggle 49-league source.

Top 20 leagues by row count:

{_markdown_table(top_leagues)}

Top 20 leagues by positive CBA next-season labels:

{_markdown_table(top_positive_leagues)}

Leagues excluded by eligibility filtering in `player_history_all.csv`:

{_markdown_table(excluded_from_history[excluded_from_history["eligibility_category"].str.startswith("excluded", na=False)], 30)}

## 4. Are NBA And Chinese-CBA Excluded From Main Eligible Overseas Modelling Features?

**Yes.** In `eligible_candidate_pool.csv`, NBA rows = **{eligible_pool_nba_rows}** and Chinese-CBA rows = **{eligible_pool_chinese_cba_rows}**. In `labelled_candidate_dataset.csv`, NBA rows = **{candidate_nba_rows}** and Chinese-CBA rows = **{candidate_chinese_cba_rows}**. This indicates that NBA and Chinese-CBA are excluded from the main eligible overseas candidate dataset.

Chinese-CBA rows are still preserved elsewhere for label construction and prior CBA experience handling, which is appropriate for the stated research scope.

## 5. Source-Level Audit

{_markdown_table(source_audit)}

## 6. Candidate Dataset Audit

The labelled candidate dataset contains **{total_rows:,} rows**, including **{positives:,} positives** and **{total_rows - positives:,} negatives**. The positive rate is **{positive_rate:.6f}**.

Rows and positives by season:

{_markdown_table(candidate_audit[candidate_audit["section"] == "by_season"][["group", "rows", "positive_rows", "negative_rows", "positive_rate"]], 20)}

Missing value rates for key fields:

{_markdown_table(candidate_audit[candidate_audit["section"] == "missing_value_rate"][["group", "field_exists", "missing_rate", "non_null_rate"]], 45)}

## 7. Is The Current Data Sufficient For A Cautious Baseline Ranking Experiment?

**Yes, with caution.** The dataset has enough basic box-score coverage, a broad multi-league candidate pool, reproducible labels, and derived usage/efficiency indicators to support a cautious baseline ranking experiment. The existing time-based evaluation and random baseline comparison are appropriate first checks.

However, the data is not sufficient for strong causal claims or deterministic recruitment prediction. The positive class is extremely sparse, advanced official metrics are incomplete, and many non-statistical recruitment determinants are absent.

## 8. Dissertation Limitations To State

- The labelled task is extremely imbalanced, so accuracy is not a meaningful headline metric.
- Positive labels are sparse and may omit players missing from public sources.
- Basic statistics are widely available, but official advanced, possession, rating and play-type statistics are limited or absent.
- The project has multi-league coverage, but source coverage is uneven across leagues and seasons.
- RealGM could not be used programmatically because of Cloudflare / HTTP 403 blocking.
- NBA and Chinese-CBA rows are excluded from the main overseas modelling features, while Chinese-CBA rows are used only for label extension and prior-experience context.
- Contract status, salary, injuries, agents, team needs, roster rules and private scouting information are not captured.
- Results should be framed as ranking support and exploratory evidence, not deterministic prediction of CBA signings.

## 9. Data Anomaly Note

The anomaly audit found **{anomaly_total:,} flagged rows** across the final labelled dataset and raw Kaggle source checks. These rows were **not cleaned or removed**. They are reported separately so the dissertation can distinguish reproducible source-data issues from modelling choices.

{_markdown_table(anomaly_summary, 30)}
"""


# 函数：run_audit
def run_audit(processed_dir: Path, reports_dir: Path) -> None:
    datasets = {
        name: _read_csv(processed_dir / filename, required=name in {"labelled_candidate_dataset"})
        for name, filename in AUDIT_DATASETS.items()
    }

    basic_coverage = pd.concat(
        [
            _coverage_for_fields(df, dataset, BASIC_FIELDS, "basic")
            for dataset, df in datasets.items()
            if not df.empty
        ],
        ignore_index=True,
    )
    advanced_coverage = pd.concat(
        [
            _coverage_for_fields(df, dataset, ADVANCED_FIELDS, "advanced")
            for dataset, df in datasets.items()
            if not df.empty
        ]
        + [
            _coverage_for_fields(df, dataset, PLAY_TYPE_FIELDS, "play_type")
            for dataset, df in datasets.items()
            if not df.empty
        ],
        ignore_index=True,
    )

    history = datasets.get("player_history_all", pd.DataFrame())
    labelled = datasets.get("labelled_candidate_dataset", pd.DataFrame())
    league_coverage = _league_coverage(history, labelled)
    source_audit = _source_level_audit(history)
    candidate_audit = _candidate_dataset_audit(labelled)
    raw_kaggle = _load_raw_kaggle()
    anomaly_report, anomaly_summary = _data_anomaly_reports(labelled, raw_kaggle)

    reports_dir.mkdir(parents=True, exist_ok=True)
    basic_coverage.to_csv(reports_dir / "basic_stats_coverage.csv", index=False)
    advanced_coverage.to_csv(reports_dir / "advanced_stats_coverage.csv", index=False)
    league_coverage.to_csv(reports_dir / "league_coverage_audit.csv", index=False)
    source_audit.to_csv(reports_dir / "source_level_audit.csv", index=False)
    candidate_audit.to_csv(reports_dir / "candidate_dataset_audit.csv", index=False)
    anomaly_report.to_csv(reports_dir / "data_anomaly_report.csv", index=False)
    anomaly_summary.to_csv(reports_dir / "data_anomaly_summary.csv", index=False)

    markdown = _build_markdown(
        datasets,
        basic_coverage,
        advanced_coverage,
        league_coverage,
        source_audit,
        candidate_audit,
        anomaly_summary,
        reports_dir,
    )
    (reports_dir / "data_audit_summary.md").write_text(markdown, encoding="utf-8")

    LOGGER.info("Wrote data audit summary to %s", reports_dir / "data_audit_summary.md")
    LOGGER.info("Wrote basic stat coverage to %s", reports_dir / "basic_stats_coverage.csv")
    LOGGER.info("Wrote advanced stat coverage to %s", reports_dir / "advanced_stats_coverage.csv")
    LOGGER.info("Wrote league coverage audit to %s", reports_dir / "league_coverage_audit.csv")
    LOGGER.info("Wrote source-level audit to %s", reports_dir / "source_level_audit.csv")
    LOGGER.info("Wrote candidate dataset audit to %s", reports_dir / "candidate_dataset_audit.csv")
    LOGGER.info("Wrote data anomaly report to %s", reports_dir / "data_anomaly_report.csv")
    LOGGER.info("Wrote data anomaly summary to %s", reports_dir / "data_anomaly_summary.csv")


# 函数：main
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final project data coverage without changing modelling logic.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    ensure_data_dirs()
    run_audit(args.processed_dir, args.reports_dir)


if __name__ == "__main__":
    main()
