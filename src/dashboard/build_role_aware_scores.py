from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPORTS_DIR = Path("data/reports")
USER_RECOMMENDATION_DIR = REPORTS_DIR / "user_recommendations"
DEFAULT_INPUT = REPORTS_DIR / "frontend_recommendations_enriched.csv"
SUMMARY_OUTPUT = REPORTS_DIR / "role_aware_scoring_summary.csv"
THRESHOLDS_OUTPUT = REPORTS_DIR / "role_scoring_thresholds.csv"
EVALUATION_OUTPUT = REPORTS_DIR / "role_aware_evaluation_summary.csv"

ROLE_SCORE_COLUMNS = [
    "score_high_usage_ball_handler",
    "score_scoring_import",
    "score_playmaking_guard",
    "score_frontcourt_import",
    "score_low_risk_availability",
]

ROLE_LABELS = {
    "score_high_usage_ball_handler": "持球核心",
    "score_scoring_import": "得分型外援",
    "score_playmaking_guard": "组织后卫",
    "score_frontcourt_import": "前场/内线外援",
    "score_low_risk_availability": "低风险稳定型候选",
}

FLAG_BY_SCORE = {
    "score_high_usage_ball_handler": "is_high_usage_candidate",
    "score_scoring_import": "is_scoring_import_candidate",
    "score_playmaking_guard": "is_playmaking_candidate",
    "score_frontcourt_import": "is_frontcourt_candidate",
    "score_low_risk_availability": "is_low_risk_candidate",
}

NUMERIC_INPUT_COLUMNS = [
    "minutes_per_game",
    "points_per_36",
    "points",
    "usage_proxy",
    "ts_pct",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "assists",
    "assists_per_36",
    "assist_to_turnover_ratio",
    "rebounds",
    "rebounds_per_36",
    "blocks",
    "blocks_per_36",
    "steals",
    "steals_per_36",
    "turnovers",
    "turnovers_per_36",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "games",
    "minutes",
    "data_completeness_score",
    "has_prior_cba_experience_before_t",
    "prior_cba_seasons_before_t",
    "height_cm",
    "height",
]

CORE_STATS = ["minutes_per_game", "points_per_36", "usage_proxy", "games", "minutes"]
EFFICIENCY_STATS = ["ts_pct", "fg_pct", "three_pct", "ft_pct"]


# 功能：把指定字段安全转换为数值。
def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


# 功能：把比例限制在合理的零到一范围内。
def _clip_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().gt(1.0).any() and values.dropna().le(100).all():
        values = values / 100.0
    return values.clip(0, 1).fillna(0.0)


# 功能：计算球员在当前候选池中的百分位排名。
def _rank_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return values.rank(pct=True, method="average").fillna(0.0).clip(0, 1)


# 功能：按指标方向选择正向或反向百分位。
def _preferred_pct(df: pd.DataFrame, raw_col: str, percentile_col: str | None = None) -> pd.Series:
    percentile_col = percentile_col or f"{raw_col}_league_season_pct"
    if percentile_col in df.columns and pd.to_numeric(df[percentile_col], errors="coerce").notna().any():
        return _clip_pct(df[percentile_col])
    return _rank_pct(_to_numeric(df, raw_col))


# 功能：安全执行除法，避免零分母和无效数值造成报错。
def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    num = pd.to_numeric(numerator, errors="coerce")
    return num / den


# 功能：用已有基础统计补算角色评分所需的缺失字段。
def _derive_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_INPUT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    minutes = _to_numeric(out, "minutes")
    games = _to_numeric(out, "games")
    if "minutes_per_game" not in out.columns or out["minutes_per_game"].isna().all():
        out["minutes_per_game"] = _safe_divide(minutes, games)
    else:
        out["minutes_per_game"] = _to_numeric(out, "minutes_per_game").fillna(_safe_divide(minutes, games))

    derivations = {
        "points_per_36": "points",
        "assists_per_36": "assists",
        "rebounds_per_36": "rebounds",
        "blocks_per_36": "blocks",
        "steals_per_36": "steals",
        "turnovers_per_36": "turnovers",
    }
    for target, source in derivations.items():
        derived = _safe_divide(_to_numeric(out, source), minutes) * 36.0
        if target in out.columns:
            out[target] = _to_numeric(out, target).fillna(derived)
        else:
            out[target] = derived

    if "shot_attempts_per_36" not in out.columns:
        shot_attempts = _to_numeric(out, "field_goal_attempts") + 0.44 * _to_numeric(out, "free_throw_attempts")
        out["shot_attempts_per_36"] = _safe_divide(shot_attempts, minutes) * 36.0
    if "three_point_attempt_rate" not in out.columns:
        out["three_point_attempt_rate"] = _safe_divide(_to_numeric(out, "three_point_attempts"), _to_numeric(out, "field_goal_attempts"))
    if "free_throw_attempt_rate" not in out.columns:
        out["free_throw_attempt_rate"] = _safe_divide(_to_numeric(out, "free_throw_attempts"), _to_numeric(out, "field_goal_attempts"))

    if "assist_to_turnover_ratio" not in out.columns:
        out["assist_to_turnover_ratio"] = _safe_divide(_to_numeric(out, "assists"), _to_numeric(out, "turnovers"))
    return out


# 功能：根据位置文字判断球员是否偏前场或内线。
def _frontcourt_position_signal(df: pd.DataFrame) -> pd.Series:
    if "position" not in df.columns:
        return pd.Series(0.0, index=df.index)
    text = df["position"].fillna("").astype(str).str.lower()
    frontcourt = text.str.contains(r"\b(?:f|pf|sf|c)\b|forward|center|centre|big", regex=True)
    return frontcourt.astype(float)


# 功能：根据身高计算前场和内线角色信号。
def _height_signal(df: pd.DataFrame) -> pd.Series:
    height = _to_numeric(df, "height_cm")
    if height.notna().sum() == 0:
        raw = df.get("height", pd.Series(index=df.index, dtype=object))
        height = pd.to_numeric(raw, errors="coerce")
        # If height looks like inches, convert to cm; otherwise leave plausible cm values as-is.
        height = height.where(~height.between(70, 90), height * 2.54)
    if height.notna().sum() == 0:
        return pd.Series(0.0, index=df.index)
    return _rank_pct(height)


# 功能：根据缺失数据、出场时间和效率生成风险标记。
def _risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    games = _to_numeric(df, "games")
    minutes = _to_numeric(df, "minutes")
    mpg = _to_numeric(df, "minutes_per_game")
    completeness = _to_numeric(df, "data_completeness_score")

    out["low_minutes_risk_flag"] = mpg.lt(12).fillna(True)
    out["low_games_risk_flag"] = games.lt(10).fillna(True)
    out["missing_efficiency_risk_flag"] = df[[c for c in EFFICIENCY_STATS if c in df.columns]].notna().sum(axis=1).eq(0) if any(c in df.columns for c in EFFICIENCY_STATS) else True
    out["missing_core_stats_risk_flag"] = df[[c for c in CORE_STATS if c in df.columns]].notna().sum(axis=1).lt(3) if any(c in df.columns for c in CORE_STATS) else True
    out["small_sample_warning"] = games.lt(10).fillna(True) | minutes.lt(250).fillna(True)
    out["data_quality_risk_flag"] = completeness.lt(0.45).fillna(out["missing_core_stats_risk_flag"])
    return out


# 功能：根据出场场次和时间计算球员可用性分数。
def _availability_score(df: pd.DataFrame, risks: pd.DataFrame) -> pd.Series:
    games_pct = _preferred_pct(df, "games")
    minutes_pct = _preferred_pct(df, "minutes")
    mpg_pct = _preferred_pct(df, "minutes_per_game")
    completeness = _to_numeric(df, "data_completeness_score").fillna(0.0).clip(0, 1)
    prior = _to_numeric(df, "has_prior_cba_experience_before_t").fillna(0).clip(0, 1)
    score = 100 * (0.30 * games_pct + 0.25 * minutes_pct + 0.20 * mpg_pct + 0.20 * completeness + 0.05 * prior)
    penalty = 8 * risks["missing_core_stats_risk_flag"].astype(float) + 6 * risks["small_sample_warning"].astype(float)
    return (score - penalty).clip(0, 100)


# 功能：计算不同外援角色画像下的适配分数。
def _score_roles(df: pd.DataFrame, risks: pd.DataFrame) -> pd.DataFrame:
    pct_points = _preferred_pct(df, "points_per_36")
    pct_usage = _preferred_pct(df, "usage_proxy")
    pct_ts = _preferred_pct(df, "ts_pct")
    pct_three = _preferred_pct(df, "three_pct")
    pct_mpg = _preferred_pct(df, "minutes_per_game")
    pct_assists = _preferred_pct(df, "assists_per_36")
    pct_ast_to = _preferred_pct(df, "assist_to_turnover_ratio")
    pct_reb = _preferred_pct(df, "rebounds_per_36")
    pct_blk = _preferred_pct(df, "blocks_per_36")
    pct_shots = _preferred_pct(df, "shot_attempts_per_36")
    pct_fta_rate = _preferred_pct(df, "free_throw_attempt_rate")
    pct_turnovers = _preferred_pct(df, "turnovers_per_36")
    pct_fg = _preferred_pct(df, "fg_pct")
    pct_height = _height_signal(df)
    pos_frontcourt = _frontcourt_position_signal(df)

    low_sample_penalty = 0.08 * risks["low_minutes_risk_flag"].astype(float) + 0.06 * risks["low_games_risk_flag"].astype(float)
    missing_eff_penalty = 0.05 * risks["missing_efficiency_risk_flag"].astype(float)
    turnover_penalty = 0.08 * pct_turnovers

    scores = pd.DataFrame(index=df.index)
    scores["score_high_usage_ball_handler"] = 100 * (
        0.28 * pct_usage + 0.24 * pct_points + 0.18 * pct_assists + 0.15 * pct_ts + 0.15 * pct_mpg
        - low_sample_penalty - turnover_penalty
    )
    scores["score_scoring_import"] = 100 * (
        0.30 * pct_points + 0.22 * pct_usage + 0.18 * pct_shots + 0.12 * pct_fta_rate + 0.12 * pct_ts + 0.06 * pct_three
        - low_sample_penalty - missing_eff_penalty
    )
    scores["score_playmaking_guard"] = 100 * (
        0.30 * pct_assists + 0.20 * pct_ast_to + 0.18 * pct_usage + 0.17 * pct_mpg + 0.15 * pct_ts
        - low_sample_penalty - turnover_penalty
    )
    scores["score_frontcourt_import"] = 100 * (
        0.28 * pct_reb + 0.20 * pct_blk + 0.16 * pct_mpg + 0.14 * pct_ts + 0.10 * pct_fg + 0.07 * pct_height + 0.05 * pos_frontcourt
        - low_sample_penalty
    )
    scores["score_low_risk_availability"] = _availability_score(df, risks)
    return scores.clip(0, 100)


# 功能：计算生成解释文字时使用的候选池参考阈值。
def _thresholds(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_col in ROLE_SCORE_COLUMNS:
        values = pd.to_numeric(scores[score_col], errors="coerce")
        if values.notna().sum() == 0:
            threshold = np.nan
        elif len(values) < 10:
            threshold = float(values.max())
        else:
            threshold = float(values.quantile(0.70))
        rows.append(
            {
                "role_score_column": score_col,
                "candidate_flag": FLAG_BY_SCORE[score_col],
                "role_label": ROLE_LABELS[score_col],
                "threshold_method": "70th percentile within selected dataset; max for very small datasets",
                "threshold_value": threshold,
                "rows_at_or_above_threshold": int(values.ge(threshold).sum()) if pd.notna(threshold) else 0,
            }
        )
    return pd.DataFrame(rows)


# 功能：根据统计、角色和路径信号生成推荐解释。
def _build_reasons(df: pd.DataFrame, scored: pd.DataFrame, risks: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    best_score_col = scored[ROLE_SCORE_COLUMNS].idxmax(axis=1)
    out["best_role_profile"] = best_score_col.map(ROLE_LABELS)
    out["best_role_score"] = scored[ROLE_SCORE_COLUMNS].max(axis=1).round(2)

    # 功能：说明球员为什么适合当前角色画像。
    def role_reason(row: pd.Series) -> str:
        parts = []
        if pd.notna(row.get("usage_proxy")):
            parts.append("使用率代理较有参考价值")
        if pd.notna(row.get("points_per_36")):
            parts.append("每36分钟得分可用于判断进攻产量")
        if pd.notna(row.get("assists_per_36")) or pd.notna(row.get("assists")):
            parts.append("组织数据可用于判断持球/传球角色")
        if pd.notna(row.get("rebounds_per_36")) or pd.notna(row.get("blocks_per_36")):
            parts.append("篮板/护筐数据可用于判断前场适配")
        return "；".join(parts) if parts else "角色判断主要受字段缺失限制，需要人工复核。"

    # 功能：说明球员的得分效率表现。
    def efficiency_reason(row: pd.Series) -> str:
        available = [c for c in ["ts_pct", "fg_pct", "three_pct", "ft_pct"] if pd.notna(row.get(c))]
        if available:
            return "有效率字段可参考：" + "、".join(available)
        return "效率字段缺失，不能仅凭得分产量判断。"

    # 功能：说明球员所在联赛与历史 CBA 招募路径的关系。
    def pathway_reason(row: pd.Series) -> str:
        league = row.get("league", "")
        source = row.get("source_id", row.get("source", ""))
        bits = []
        if pd.notna(league) and str(league).strip():
            bits.append(f"联赛：{league}")
        if pd.notna(source) and str(source).strip():
            bits.append(f"来源：{source}")
        if pd.to_numeric(pd.Series([row.get("has_prior_cba_experience_before_t")]), errors="coerce").fillna(0).iloc[0] > 0:
            bits.append("有过往 CBA 经历")
        return "；".join(bits) if bits else "路径信息有限。"

    # 功能：说明球员的出勤和上场时间情况。
    def availability_reason(row: pd.Series) -> str:
        games = row.get("games")
        mpg = row.get("minutes_per_game")
        pieces = []
        if pd.notna(games):
            pieces.append(f"出场 {games:g} 场")
        if pd.notna(mpg):
            pieces.append(f"场均 {mpg:.1f} 分钟")
        if pd.notna(row.get("data_completeness_score")):
            pieces.append(f"数据完整度 {float(row.get('data_completeness_score')):.2f}")
        return "；".join(pieces) if pieces else "可用性字段不足。"

    # 功能：汇总该球员需要人工核查的风险。
    def risk_summary(idx: int) -> str:
        flags = []
        if bool(risks.loc[idx, "low_minutes_risk_flag"]):
            flags.append("上场时间偏低")
        if bool(risks.loc[idx, "low_games_risk_flag"]):
            flags.append("样本场次偏少")
        if bool(risks.loc[idx, "missing_efficiency_risk_flag"]):
            flags.append("效率字段缺失")
        if bool(risks.loc[idx, "missing_core_stats_risk_flag"]):
            flags.append("核心统计字段缺失")
        return "；".join(flags) if flags else "未发现主要数据风险"

    out["role_fit_reason"] = df.apply(role_reason, axis=1)
    out["efficiency_reason"] = df.apply(efficiency_reason, axis=1)
    out["pathway_reason"] = df.apply(pathway_reason, axis=1)
    out["availability_reason"] = df.apply(availability_reason, axis=1)
    out["risk_summary"] = [risk_summary(idx) for idx in df.index]
    out["final_reason_summary"] = (
        out["best_role_profile"].fillna("角色待定")
        + "；"
        + out["role_fit_reason"].fillna("")
        + "；风险提示："
        + out["risk_summary"].fillna("")
    )
    return out


# 功能：为候选人添加角色适配分、解释文字和风险提示。
def add_role_aware_scores(df: pd.DataFrame, *, write_reports: bool = False, source_label: str = "selected_dataset") -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    base = _derive_missing_columns(df)
    risks = _risk_flags(base)
    scores = _score_roles(base, risks)
    thresholds = _thresholds(scores)

    out = base.copy()
    for col in ROLE_SCORE_COLUMNS:
        out[col] = scores[col].round(2)
    out["availability_score"] = scores["score_low_risk_availability"].round(2)
    for col in risks.columns:
        out[col] = risks[col]

    for _, row in thresholds.iterrows():
        score_col = row["role_score_column"]
        flag_col = row["candidate_flag"]
        threshold_value = row["threshold_value"]
        out[flag_col] = scores[score_col].ge(threshold_value) if pd.notna(threshold_value) else False

    reasons = _build_reasons(base, scores, risks)
    for col in reasons.columns:
        out[col] = reasons[col]

    if write_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        thresholds.insert(0, "source_label", source_label)
        thresholds.to_csv(THRESHOLDS_OUTPUT, index=False, encoding="utf-8-sig")
        _summary(out, source_label).to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
        evaluation = _optional_evaluation(out)
        if not evaluation.empty:
            evaluation.to_csv(EVALUATION_OUTPUT, index=False, encoding="utf-8-sig")
    return out


# 功能：汇总当前处理结果的关键数量和比例。
def _summary(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    rows = [{"metric": "source_label", "value": source_label}, {"metric": "rows", "value": len(df)}]
    for col in ROLE_SCORE_COLUMNS + ["availability_score"]:
        if col in df.columns:
            rows.extend(
                [
                    {"metric": f"{col}_mean", "value": float(pd.to_numeric(df[col], errors="coerce").mean())},
                    {"metric": f"{col}_non_null", "value": int(pd.to_numeric(df[col], errors="coerce").notna().sum())},
                ]
            )
    if "best_role_profile" in df.columns:
        for role, count in df["best_role_profile"].value_counts(dropna=False).items():
            rows.append({"metric": f"best_role_profile_{role}", "value": int(count)})
    for col in ["data_quality_risk_flag", "low_minutes_risk_flag", "low_games_risk_flag", "missing_efficiency_risk_flag", "missing_core_stats_risk_flag", "small_sample_warning"]:
        if col in df.columns:
            rows.append({"metric": f"{col}_count", "value": int(df[col].fillna(False).astype(bool).sum())})
    return pd.DataFrame(rows)


# 功能：计算前 K 名候选中的真实正例比例。
def _precision_at_k(y: pd.Series, k: int) -> float:
    top = y.head(min(k, len(y)))
    return float(top.mean()) if len(top) else 0.0


# 功能：计算前 K 名候选覆盖了多少真实正例。
def _recall_at_k(y: pd.Series, k: int) -> float:
    positives = y.sum()
    if positives <= 0:
        return 0.0
    return float(y.head(min(k, len(y))).sum() / positives)


# 功能：在存在真实标签时计算可选的历史评估指标。
def _optional_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    if "signed_cba_next_season" not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df["signed_cba_next_season"], errors="coerce").fillna(0).astype(int)
    if y.sum() == 0:
        return pd.DataFrame()
    candidates = []
    if "rank" in df.columns:
        candidates.append(("original_rank", df.sort_values("rank")))
    elif "new_rank" in df.columns:
        candidates.append(("original_rank", df.sort_values("new_rank")))
    for score_col in ROLE_SCORE_COLUMNS:
        candidates.append((score_col, df.assign(_score=pd.to_numeric(df[score_col], errors="coerce").fillna(-1)).sort_values("_score", ascending=False)))
    rows = []
    base_rate = float(y.mean()) if len(y) else 0.0
    for name, ranked in candidates:
        ranked_y = pd.to_numeric(ranked["signed_cba_next_season"], errors="coerce").fillna(0).astype(int)
        p20 = _precision_at_k(ranked_y, 20)
        rows.append(
            {
                "ranking": name,
                "rows": len(ranked),
                "positives": int(ranked_y.sum()),
                "Precision@20": p20,
                "Precision@50": _precision_at_k(ranked_y, 50),
                "Recall@100": _recall_at_k(ranked_y, 100),
                "Lift@20": p20 / base_rate if base_rate > 0 else np.nan,
                "note": "exploratory role-aware ranking only; does not replace final dissertation result",
            }
        )
    return pd.DataFrame(rows)


# 功能：读取推荐文件并补充角色评分和解释字段。
def enrich_file(input_path: str | Path, output_path: str | Path | None = None, *, write_reports: bool = True) -> pd.DataFrame:
    input_path = Path(input_path)
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    enriched = add_role_aware_scores(df, write_reports=write_reports, source_label=str(input_path))
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    return enriched


# 功能：执行角色画像评分和解释字段生成流程。
def main() -> None:
    parser = argparse.ArgumentParser(description="Add transparent role-aware scouting scores to recommendation files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Recommendation CSV to enrich.")
    parser.add_argument("--output", default="", help="Optional output CSV. If omitted, reports only are written.")
    parser.add_argument("--no-reports", action="store_true", help="Do not write scoring summary/threshold reports.")
    args = parser.parse_args()
    output = args.output or None
    enriched = enrich_file(args.input, output, write_reports=not args.no_reports)
    print(f"Rows scored: {len(enriched)}")
    if output:
        print(f"Wrote role-aware recommendations to {output}")
    print(f"Wrote role scoring summary to {SUMMARY_OUTPUT}")
    print(f"Wrote role scoring thresholds to {THRESHOLDS_OUTPUT}")


if __name__ == "__main__":
    main()
