"""
Module: app.py
Purpose: Streamlit dashboard entrypoint for candidate recommendation visualization, bilingual UI, data-source switching, and scouting workflow interactions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from src.dashboard.ingest_new_candidates import REQUIRED_COLUMNS, ingest_candidates
from src.dashboard.parse_pasted_candidate_text import parse_pasted_text
from src.dashboard.rank_new_candidates import rank_candidates
from src.dashboard.build_role_aware_scores import ROLE_SCORE_COLUMNS, add_role_aware_scores
from src.dashboard.dashboard_i18n import field_label, t, translate_phrase_text
from src.dashboard.recommendation_source_registry import (
    DEFAULT_DISPLAY_NAME,
    delete_source,
    ensure_registry,
    is_user_source,
    list_sources,
    load_selected_recommendations,
    save_generated_recommendation_source,
)

ENRICHED_DATA_PATH = Path("data/reports/frontend_recommendations_enriched.csv")
BASE_DATA_PATH = Path("data/reports/frontend_recommendations.csv")
UPLOAD_DIR = Path("data/manual/uploads")
NEW_CANDIDATE_OUTPUT = Path("data/reports/new_candidate_recommendations.csv")

HIDDEN_URL_COLUMNS = {
    "youtube_search_url",
    "google_video_search_url",
    "youtube_highlight_search_url",
    "video_search_query",
    "cba_video_search_query",
    "official_player_url",
    "official_stats_url",
    "official_search_url",
    "nba_stats_url",
    "euroleague_profile_or_search_url",
    "league_official_search_url",
    "basketball_reference_search_url",
}

PERCENT_COLUMNS = ["ts_pct", "efg_pct", "fg_pct", "three_pct", "ft_pct"]
NUMERIC_COLUMNS = [
    "rank",
    "score",
    "games",
    "minutes",
    "minutes_per_game",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "points_per_36",
    "usage_proxy",
    "field_goal_attempts",
    "three_point_attempts",
    "free_throw_attempts",
    "data_completeness_score",
    "has_prior_cba_experience_before_t",
    "prior_cba_seasons_before_t",
    "prior_cba_last_seen_gap",
    "points_per_36_trend",
    "usage_proxy_trend",
    "ts_pct_trend",
    "minutes_per_game_trend",
    "height",
    "weight",
    "age",
    "age_at_recommendation_season",
    "best_role_score",
    "availability_score",
    "score_high_usage_ball_handler",
    "score_scoring_import",
    "score_playmaking_guard",
    "score_frontcourt_import",
    "score_low_risk_availability",
]

ROLE_PROFILE_OPTIONS = {
    "all": {"score": None, "flag": None},
    "high_usage_ball_handler": {"score": "score_high_usage_ball_handler", "flag": "is_high_usage_candidate"},
    "scoring_import": {"score": "score_scoring_import", "flag": "is_scoring_import_candidate"},
    "playmaking_guard": {"score": "score_playmaking_guard", "flag": "is_playmaking_candidate"},
    "frontcourt_import": {"score": "score_frontcourt_import", "flag": "is_frontcourt_candidate"},
    "low_risk_availability": {"score": "score_low_risk_availability", "flag": "is_low_risk_candidate"},
}

TABLE_COLUMNS = [
    "rank",
    "player_name_raw",
    "recommendation_season",
    "league",
    "team",
    "source",
    "score",
    "minutes_per_game",
    "points_per_36",
    "usage_proxy",
    "ts_pct",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "height",
    "weight",
    "age_at_recommendation_season",
    "age",
    "position",
    "country",
    "has_prior_cba_experience_before_t",
    "best_role_profile",
    "best_role_score",
    "role_fit_reason",
    "risk_summary",
    "availability_score",
    "score_high_usage_ball_handler",
    "score_scoring_import",
    "score_playmaking_guard",
    "score_frontcourt_import",
    "score_low_risk_availability",
    "reason_summary",
]


# 功能：读取当前选中的推荐数据，并在缺少正式文件时使用演示数据。
def load_data(display_name: str | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    df, source_meta = load_selected_recommendations(display_name)
    if df.empty:
        return df, source_meta
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_role_aware_scores(df, write_reports=False, source_label=str(source_meta.get("display_name", "dashboard_source")))
    for col in ROLE_SCORE_COLUMNS + ["best_role_score", "availability_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rank" in df.columns:
        df = df.sort_values("rank")
    return df, source_meta




# 功能：安全取得数据列；缺失时返回同长度的空列。
def safe_series(df: pd.DataFrame, col: str, default: object = 0) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


# 功能：安全取得数值列，并把无法解析的内容设为空值。
def safe_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(safe_series(df, col, default), errors="coerce").fillna(default)


# 功能：安全统计布尔条件为真的记录数量。
def safe_bool_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(safe_numeric_series(df, col, 0.0).gt(0).sum())


# 功能：把空值转换成适合界面显示的内容。
def format_missing_safe(value: object) -> str:
    try:
        if pd.isna(value) or value == "":
            return "-"
    except (TypeError, ValueError):
        return "-"
    return str(value)


# 功能：从多个候选字段中取第一个可用值。
def _first_available_value(df: pd.DataFrame, columns: list[str]) -> object:
    for col in columns:
        if col in df.columns and df[col].notna().any():
            values = df[col].dropna().astype(str)
            values = values[~values.str.strip().isin(["", "nan", "None"])]
            if not values.empty:
                return values.iloc[0]
    return "-"

# 功能：把单个值整理成适合 Dashboard 显示的文本。
def _display_value(value: object) -> str:
    if pd.isna(value) or value == "":
        return "-"
    return str(value)


# 功能：把数值格式化成百分比文本。
def _format_percent(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "-"
    return f"{number:.1%}" if 0 <= number <= 1 else f"{number:.2f}"


# 功能：按指定小数位格式化数值。
def _format_number(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "-"
    return f"{number:.2f}"


# 功能：读取当前 Dashboard 选择的界面语言。
def _dashboard_lang() -> str:
    return str(st.session_state.get("language", "zh"))


# 功能：把真假值转换成当前语言下的可读文本。
def _format_bool(value: object, lang: str | None = None) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "-"
    return t("yes", lang or _dashboard_lang()) if number > 0 else t("no", lang or _dashboard_lang())


# 功能：把原始字段名转换成当前语言下的友好标签。
def display_field_label(field_name: str, lang: str) -> str:
    key = f"field_{field_name}"
    translated = t(key, lang)
    if translated != key:
        return translated
    translated = field_label(field_name, lang)
    if translated != field_name:
        return translated
    cleaned = str(field_name).replace("_", " ").strip()
    return cleaned.title() if lang == "en" else cleaned


ROLE_VALUE_ALIASES = {
    "high_usage_ball_handler": "high_usage_ball_handler",
    "scoring_import": "scoring_import",
    "playmaking_guard": "playmaking_guard",
    "frontcourt_import": "frontcourt_import",
    "low_risk_availability": "low_risk_availability",
    "\u6301\u7403\u6838\u5fc3": "high_usage_ball_handler",
    "\u5f97\u5206\u578b\u5916\u63f4": "scoring_import",
    "\u7ec4\u7ec7\u540e\u536b": "playmaking_guard",
    "\u524d\u573a/\u5185\u7ebf\u5916\u63f4": "frontcourt_import",
    "\u4f4e\u98ce\u9669\u7a33\u5b9a\u578b\u5019\u9009": "low_risk_availability",
}


# 功能：把内部角色类型转换成当前语言下的角色名称。
def display_role_profile_value(value: object, lang: str) -> str:
    text = format_missing_safe(value)
    role_key = ROLE_VALUE_ALIASES.get(text)
    if role_key:
        return t(f"role_{role_key}", lang)
    return text


ROLE_REASON_FIELDS = {
    "best_role_profile",
    "role_fit_reason",
    "efficiency_reason",
    "pathway_reason",
    "availability_reason",
    "risk_summary",
    "final_reason_summary",
}


# 功能：从候选人记录中安全读取一个数值。
def _numeric_value(row: pd.Series, field: str) -> float | None:
    if field not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


# 功能：判断候选人记录中的字段是否含有效数值。
def _has_numeric_value(row: pd.Series, field: str) -> bool:
    return _numeric_value(row, field) is not None


# 功能：格式化推荐理由中需要展示的数字。
def _format_reason_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


# 功能：生成英文角色适配说明。
def _english_role_fit_reason(row: pd.Series) -> str:
    parts: list[str] = []
    if _has_numeric_value(row, "usage_proxy"):
        parts.append("Usage proxy is available as a useful role indicator")
    if _has_numeric_value(row, "points_per_36"):
        parts.append("Points per 36 can be used to assess scoring volume")
    if _has_numeric_value(row, "assists_per_36") or _has_numeric_value(row, "assists"):
        parts.append("Playmaking indicators are available")
    position = str(row.get("position", "") or "").lower()
    if _has_numeric_value(row, "rebounds_per_36") or _has_numeric_value(row, "blocks_per_36") or any(token in position for token in ["f", "c", "frontcourt", "forward", "center", "centre"]):
        parts.append("Frontcourt indicators are available")
    return "; ".join(parts) if parts else "Role indicators are limited or missing"


# 功能：生成英文效率表现说明。
def _english_efficiency_reason(row: pd.Series) -> str:
    if _has_numeric_value(row, "ts_pct"):
        return "Efficiency field available: TS%"
    if any(_has_numeric_value(row, field) for field in ["fg_pct", "three_pct", "ft_pct"]):
        return "Shooting percentage fields are available"
    return "Efficiency fields are limited or missing"


# 功能：生成英文联赛路径说明。
def _english_pathway_reason(row: pd.Series) -> str:
    league = format_missing_safe(row.get("league", "-"))
    source = format_missing_safe(row.get("source_id", row.get("source", "-")))
    return f"League: {league}; Source: {source}"


# 功能：生成英文出勤和可用性说明。
def _english_availability_reason(row: pd.Series) -> str:
    parts: list[str] = []
    mpg = _numeric_value(row, "minutes_per_game")
    completeness = _numeric_value(row, "data_completeness_score")
    games = _numeric_value(row, "games")
    if mpg is not None:
        parts.append(f"Minutes per game: {_format_reason_number(mpg, 1)}")
    if completeness is not None:
        parts.append(f"Data completeness: {_format_reason_number(completeness, 2)}")
    if games is not None:
        parts.append(f"Games: {_format_reason_number(games, 0)}")
    return "; ".join(parts) if parts else "Availability fields are limited or missing"


# 功能：生成英文风险提示。
def _english_risk_summary(row: pd.Series, value: object) -> str:
    text = translate_phrase_text(format_missing_safe(value), "en")
    if text in {"-", "", "nan", "None"}:
        return "No major data-quality risk detected"
    if "?" not in text and "??" not in text and "??" not in text:
        return text
    translated = translate_phrase_text(text, "en")
    return translated if translated != text else "No major data-quality risk detected"


# 功能：汇总生成英文推荐理由。
def _english_final_reason_summary(row: pd.Series) -> str:
    role = display_role_profile_value(row.get("best_role_profile", ""), "en")
    role_reason = _english_role_fit_reason(row)
    risk = _english_risk_summary(row, row.get("risk_summary", ""))
    return f"{role}; {role_reason}; risk note: {risk[:1].lower() + risk[1:] if risk else risk}"


# 功能：按界面语言整理角色评分和推荐理由。
def format_role_reason_value(field_name: str, value: object, row: pd.Series, lang: str) -> str:
    if field_name == "best_role_profile":
        return display_role_profile_value(value, lang)
    if field_name not in ROLE_REASON_FIELDS:
        return format_missing_safe(value)
    if lang == "zh":
        return translate_phrase_text(format_missing_safe(value), "zh")
    if field_name == "role_fit_reason":
        return _english_role_fit_reason(row)
    if field_name == "efficiency_reason":
        return _english_efficiency_reason(row)
    if field_name == "pathway_reason":
        return _english_pathway_reason(row)
    if field_name == "availability_reason":
        return _english_availability_reason(row)
    if field_name == "risk_summary":
        return _english_risk_summary(row, value)
    if field_name == "final_reason_summary":
        return _english_final_reason_summary(row)
    return translate_phrase_text(format_missing_safe(value), lang)






STREAMLIT_DISPLAY_NUMERIC_COLUMNS = {
    "rank",
    "new_rank",
    "score",
    "recommendation_score",
    "minutes_per_game",
    "points_per_36",
    "usage_proxy",
    "ts_pct",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "age",
    "games",
    "minutes",
    "has_prior_cba_experience_before_t",
    "prior_cba_seasons_before_t",
    "prior_cba_last_seen_gap",
    "best_role_score",
    "availability_score",
    "score_high_usage_ball_handler",
    "score_scoring_import",
    "score_playmaking_guard",
    "score_frontcourt_import",
    "score_low_risk_availability",
}


# 功能：复制并整理表格字段类型，避免 Streamlit 显示报错。
def sanitize_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in STREAMLIT_DISPLAY_NUMERIC_COLUMNS:
            converted = pd.to_numeric(out[col], errors="coerce")
            # Formatted display tables may contain values such as "58.9%" or "-".
            # Keep those as strings rather than replacing the visible value with NaN.
            if converted.notna().any() or out[col].isna().all():
                out[col] = converted
            else:
                out[col] = out[col].fillna("").astype(str)
        elif pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].fillna("").astype(str)
    return out

# 功能：把表格整理成适合 Streamlit 展示的格式。
def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    shown = df[[c for c in TABLE_COLUMNS if c in df.columns and c not in HIDDEN_URL_COLUMNS]].copy()
    for col in shown.columns:
        if col in PERCENT_COLUMNS:
            shown[col] = shown[col].map(_format_percent)
        elif col == "has_prior_cba_experience_before_t":
            shown[col] = shown[col].map(_format_bool)
        elif col in NUMERIC_COLUMNS and col != "rank":
            shown[col] = shown[col].map(_format_number)
        else:
            shown[col] = shown[col].map(_display_value)
    return shown




# 功能：根据当前数据量生成不会越界的 Top N 选择器。
def safe_top_n_selector(df: pd.DataFrame, default_options: list[int] | None = None, lang: str | None = None) -> int:
    lang = lang or _dashboard_lang()
    default_options = default_options or [20, 50, 100, 300]
    row_count = len(df)
    if row_count <= 0:
        st.sidebar.caption(t("no_candidates_source", lang))
        return 0

    if row_count < 20:
        options = list(range(1, row_count + 1))
        return int(st.sidebar.selectbox(t("top_n", lang), options, index=len(options) - 1))

    options = [x for x in default_options if x <= row_count]
    if row_count < max(default_options) and row_count not in options:
        options.append(row_count)
    options = sorted(set(options)) or [row_count]
    default_value = row_count if row_count in options and row_count < 100 else min(100, max(options))
    default_index = options.index(default_value) if default_value in options else len(options) - 1
    return int(st.sidebar.selectbox(t("top_n", lang), options, index=default_index))

# 功能：按照所选角色画像对应的分数重新排序。
def _sort_for_role_profile(df: pd.DataFrame, role_profile: str) -> pd.DataFrame:
    config = ROLE_PROFILE_OPTIONS.get(role_profile, ROLE_PROFILE_OPTIONS["all"])
    score_col = config.get("score")
    flag_col = config.get("flag")
    out = df.copy()
    if score_col and score_col in out.columns:
        if flag_col and flag_col in out.columns:
            flagged = out[out[flag_col].fillna(False).astype(bool)].copy()
            if not flagged.empty:
                out = flagged
        out["_role_sort_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(-1)
        sort_cols = ["_role_sort_score"]
        ascending = [False]
        if "rank" in out.columns:
            sort_cols.append("rank")
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).drop(columns=["_role_sort_score"], errors="ignore")
    if "rank" in out.columns:
        return out.sort_values("rank")
    return out


# 功能：应用联赛、来源、出场时间和姓名等侧边栏筛选条件。
def filtered_data(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    st.sidebar.header(t("filters", lang))
    top_n = safe_top_n_selector(df, lang=lang)

    role_keys = list(ROLE_PROFILE_OPTIONS.keys())
    role_labels = {key: t(f"role_{key}", lang) for key in role_keys}
    selected_role_label = st.sidebar.selectbox(t("role_profile", lang), [role_labels[key] for key in role_keys])
    role_profile = next(key for key, label in role_labels.items() if label == selected_role_label)
    all_label = t("all", lang)
    leagues = [all_label] + sorted(df["league"].dropna().astype(str).unique().tolist()) if "league" in df else [all_label]
    league = st.sidebar.selectbox(t("league", lang), leagues)
    prior = "all"
    if "has_prior_cba_experience_before_t" in df.columns:
        prior_options = {
            "all": t("all", lang),
            "prior_only": t("prior_cba_only", lang),
            "no_prior": t("no_prior_cba", lang),
        }
        prior_label = st.sidebar.selectbox(t("prior_cba_experience", lang), list(prior_options.values()))
        prior = next(key for key, label in prior_options.items() if label == prior_label)
    else:
        st.sidebar.caption(t("no_cba_context_field", lang))

    min_minutes = 0.0
    minutes = pd.to_numeric(df["minutes_per_game"], errors="coerce") if "minutes_per_game" in df.columns else pd.Series(dtype=float)
    valid_minutes = minutes.dropna()
    if not valid_minutes.empty:
        valid_min = float(valid_minutes.min())
        valid_max = float(valid_minutes.max())
        if valid_max > valid_min and valid_max > 0:
            step = 0.5 if valid_max >= 0.5 else max(valid_max / 10, 0.01)
            min_minutes = st.sidebar.slider(t("minimum_minutes_per_game", lang), 0.0, valid_max, 0.0, step)
        else:
            st.sidebar.caption(t("no_minutes_range", lang))
    else:
        st.sidebar.caption(t("no_minutes_range", lang))
    search = st.sidebar.text_input(t("search_by_player_name", lang), "")

    out = df.copy()
    if league != all_label and "league" in out.columns:
        out = out[out["league"].astype(str).eq(league)]
    if prior == "prior_only":
        out = out[safe_numeric_series(out, "has_prior_cba_experience_before_t", 0.0).gt(0)]
    elif prior == "no_prior":
        out = out[~safe_numeric_series(out, "has_prior_cba_experience_before_t", 0.0).gt(0)]
    if min_minutes and "minutes_per_game" in out.columns:
        out = out[pd.to_numeric(out["minutes_per_game"], errors="coerce").fillna(0).ge(min_minutes)]
    if search.strip() and "player_name_raw" in out.columns:
        out = out[out["player_name_raw"].astype(str).str.contains(search.strip(), case=False, na=False)]

    out = _sort_for_role_profile(out, role_profile)
    if top_n > 0:
        out = out.head(top_n)
    return out
# 功能：显示一个可安全打开外部网页的按钮。
def _link_button(label: str, url: object, lang: str | None = None) -> None:
    url_text = str(url or "").strip()
    if not url_text or url_text.lower() in {"nan", "none"}:
        st.caption(f"{label}: {t('unavailable', lang or _dashboard_lang())}")
        return
    if hasattr(st, "link_button"):
        st.link_button(label, url_text)
    else:
        st.markdown(f"[{label}]({url_text})")


# 功能：生成球员的 Basketball-Reference 手动搜索链接。
def _basketball_reference_search_url(player: pd.Series) -> str:
    existing = player.get("basketball_reference_search_url")
    if pd.notna(existing) and str(existing).strip():
        return str(existing)
    query = f'site:basketball-reference.com "{player.get("player_name_raw", "")}" basketball'
    return "https://www.google.com/search?q=" + quote_plus(query)




# 功能：读取当前推荐数据源的登记信息。
def _selected_source_metadata(source_meta: dict[str, object]) -> pd.DataFrame:
    keys = ["display_name", "source_type", "season", "league", "row_count", "created_at", "notes"]
    rows = [{"field": key, "value": _display_value(source_meta.get(key, ""))} for key in keys]
    return pd.DataFrame(rows)


# 功能：取得默认推荐数据源的显示名称。
def _default_display_name(prefix: str) -> str:
    return f"{prefix} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H%M')}"

# 功能：整理主推荐名单需要展示的字段。
def candidate_table(df: pd.DataFrame) -> None:
    st.dataframe(sanitize_dataframe_for_streamlit(_format_table(df)), width="stretch", hide_index=True)


# 功能：整理用户新数据推荐名单需要展示的字段。
def new_candidate_table(df: pd.DataFrame) -> None:
    cols = [
        "new_rank",
        "player_name_raw",
        "season",
        "league",
        "team",
        "source",
        "recommendation_score",
        "minutes_per_game",
        "points_per_36",
        "usage_proxy",
        "ts_pct",
        "fg_pct",
        "three_pct",
        "ft_pct",
        "height",
        "weight",
        "age",
        "best_role_profile",
        "best_role_score",
        "role_fit_reason",
        "risk_summary",
        "availability_score",
        "score_high_usage_ball_handler",
        "score_scoring_import",
        "score_playmaking_guard",
        "score_frontcourt_import",
        "score_low_risk_availability",
        "reason_summary",
        "data_quality_warning",
    ]
    shown = df[[c for c in cols if c in df.columns]].copy()
    for col in shown.columns:
        if col in PERCENT_COLUMNS:
            shown[col] = shown[col].map(_format_percent)
        elif col in NUMERIC_COLUMNS or col in {"new_rank", "recommendation_score"}:
            shown[col] = shown[col].map(_format_number)
        else:
            shown[col] = shown[col].map(_display_value)
    st.dataframe(sanitize_dataframe_for_streamlit(shown), width="stretch", hide_index=True)

# 功能：生成球员人工搜索与核查链接表。
def _search_links_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rank", "player_name_raw", "league", "team", "video_search_query", "youtube_search_url", "google_video_search_url"]
    shown = df[[c for c in cols if c in df.columns]].copy()
    return shown






# 功能：生成侧边栏数据源选项的显示文本。
def _sidebar_display_name(name: object, max_length: int = 48) -> str:
    text = format_missing_safe(name)
    return text if len(text) <= max_length else text[: max_length - 1] + "..."

# 功能：显示数据源信息并提供受保护的删除操作。
def _source_management_panel(source_meta: dict[str, object], sources: pd.DataFrame, lang: str) -> None:
    with st.sidebar.expander(t("manage_sources", lang), expanded=False):
        rows = []
        for key in ["display_name", "source_type", "row_count", "created_at"]:
            rows.append({"field": field_label(key, lang), "value": _display_value(source_meta.get(key, ""))})
        st.table(sanitize_dataframe_for_streamlit(pd.DataFrame(rows)))

        if not is_user_source(source_meta):
            st.info(t("default_not_deletable", lang))
            return

        delete_files = st.checkbox(t("delete_local_files", lang), value=True)
        confirmation = st.text_input(t("type_delete", lang), value="", key=f"delete_confirm_{source_meta.get('source_id', '')}")
        if st.button(t("delete_current_source", lang), key=f"delete_source_{source_meta.get('source_id', '')}"):
            if confirmation != "DELETE":
                st.warning(t("type_delete_warning", lang))
                return
            result = delete_source(str(source_meta.get("source_id", "")), delete_files=delete_files)
            if result.get("error"):
                st.error(str(result["error"]))
                return
            st.success(t("source_deleted", lang, name=result.get("deleted_display_name", "")))
            st.rerun()


# 功能：整理当前可选的数据源清单。
def _available_sources_table(sources: pd.DataFrame) -> pd.DataFrame:
    cols = ["display_name", "source_type", "row_count", "created_at", "recommendation_file"]
    return sources[[c for c in cols if c in sources.columns]].copy()


# 功能：生成用于查找指定球员的搜索关键词。
def _search_query(player: pd.Series) -> str:
    name = str(player.get("player_name_raw", "") or "").strip()
    league = str(player.get("league", "") or "").strip()
    query = f"{name} {league} basketball highlights".strip()
    return query or f"{name} basketball highlights".strip()


# 功能：生成球员的 YouTube 手动搜索链接。
def _youtube_search_url(player: pd.Series) -> str:
    existing = player.get("youtube_search_url")
    if pd.notna(existing) and str(existing).strip():
        return str(existing)
    return "https://www.youtube.com/results?search_query=" + quote_plus(_search_query(player))


# 功能：生成球员的 Google 视频搜索链接。
def _google_video_search_url(player: pd.Series) -> str:
    existing = player.get("google_video_search_url")
    if pd.notna(existing) and str(existing).strip():
        return str(existing)
    return "https://www.google.com/search?tbm=vid&q=" + quote_plus(_search_query(player))


# 功能：生成球员所在联赛或球队的官方信息搜索链接。
def _league_official_search_url(player: pd.Series) -> str:
    for col in ["league_official_search_url", "official_search_url"]:
        existing = player.get(col)
        if pd.notna(existing) and str(existing).strip():
            return str(existing)
    name = str(player.get("player_name_raw", "") or "").strip()
    league = str(player.get("league", "") or "").strip()
    query = f'"{name}" "{league}" official basketball stats'
    return "https://www.google.com/search?q=" + quote_plus(query)


# 功能：从候选人记录中读取用于摘要展示的值。
def _summary_value(row: pd.Series, fields: list[str]) -> object:
    for field in fields:
        if field in row.index:
            value = row.get(field)
            if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
                return value
    return "-"


# 功能：把候选人字段整理成详情面板的字段—内容列表。
def _detail_rows(row: pd.Series, fields: list[tuple[str, str]], percent_fields: set[str] | None = None, bool_fields: set[str] | None = None, lang: str | None = None) -> pd.DataFrame:
    lang = lang or _dashboard_lang()
    percent_fields = percent_fields or set()
    bool_fields = bool_fields or set()
    rows = []
    for label, field in fields:
        value = row.get(field, pd.NA) if field in row.index else pd.NA
        if field in percent_fields:
            shown = _format_percent(value)
        elif field in bool_fields:
            shown = _format_bool(value, lang)
        elif field in NUMERIC_COLUMNS or field in {"new_rank", "recommendation_score"}:
            shown = _format_number(value)
        else:
            shown = format_role_reason_value(field, value, row, lang) if field in ROLE_REASON_FIELDS else format_missing_safe(value)
        rows.append({display_field_label("field", lang): display_field_label(label, lang), display_field_label("value", lang): shown})
    return pd.DataFrame(rows)


# 功能：显示候选人的表现、背景、角色、风险和核查链接。
def render_candidate_detail(row: pd.Series, selected_source_metadata: dict[str, object] | None = None, lang: str | None = None) -> None:
    lang = lang or _dashboard_lang()
    st.subheader(str(_summary_value(row, ["player_name_raw"])))

    st.markdown(f"**{t('candidate_summary', lang)}**")
    summary_fields = [
        ("player_name_raw", "player_name_raw"),
        ("rank", "rank"),
        ("new_rank", "new_rank"),
        ("score", "score"),
        ("recommendation_score", "recommendation_score"),
        ("season", "season"),
        ("recommendation_season", "recommendation_season"),
        ("league", "league"),
        ("team", "team"),
        ("source", "source"),
        ("reason_summary", "reason_summary"),
    ]
    summary = _detail_rows(row, summary_fields, lang=lang)
    st.table(sanitize_dataframe_for_streamlit(summary))

    st.markdown(f"**{t('key_performance', lang)}**")
    performance_fields = [
        ("minutes_per_game", "minutes_per_game"),
        ("points_per_36", "points_per_36"),
        ("usage_proxy", "usage_proxy"),
        ("ts_pct", "ts_pct"),
        ("efg_pct", "efg_pct"),
        ("fg_pct", "fg_pct"),
        ("three_pct", "three_pct"),
        ("ft_pct", "ft_pct"),
        ("games", "games"),
        ("minutes", "minutes"),
        ("rebounds", "rebounds"),
        ("rebounds_per_36", "rebounds_per_36"),
        ("assists", "assists"),
        ("assists_per_36", "assists_per_36"),
        ("turnovers", "turnovers"),
        ("turnovers_per_36", "turnovers_per_36"),
    ]
    st.table(sanitize_dataframe_for_streamlit(_detail_rows(row, performance_fields, percent_fields=set(PERCENT_COLUMNS + ["efg_pct"]), lang=lang)))

    st.markdown(f"**{t('role_risk', lang)}**")
    role_fields = [
        ("best_role_profile", "best_role_profile"),
        ("best_role_score", "best_role_score"),
        ("availability_score", "availability_score"),
        ("score_high_usage_ball_handler", "score_high_usage_ball_handler"),
        ("score_scoring_import", "score_scoring_import"),
        ("score_playmaking_guard", "score_playmaking_guard"),
        ("score_frontcourt_import", "score_frontcourt_import"),
        ("score_low_risk_availability", "score_low_risk_availability"),
        ("role_fit_reason", "role_fit_reason"),
        ("efficiency_reason", "efficiency_reason"),
        ("pathway_reason", "pathway_reason"),
        ("availability_reason", "availability_reason"),
        ("risk_summary", "risk_summary"),
        ("final_reason_summary", "final_reason_summary"),
    ]
    st.table(sanitize_dataframe_for_streamlit(_detail_rows(row, role_fields, lang=lang)))
    st.markdown(f"**{t('biodata_context', lang)}**")
    biodata_fields = [
        ("height", "height"),
        ("weight", "weight"),
        ("birthdate", "birthdate"),
        ("age_at_recommendation_season", "age_at_recommendation_season"),
        ("age", "age"),
        ("position", "position"),
        ("country", "country"),
        ("last_affiliation", "last_affiliation"),
    ]
    st.table(sanitize_dataframe_for_streamlit(_detail_rows(row, biodata_fields, lang=lang)))

    st.markdown(f"**{t('cba_context', lang)}**")
    cba_fields = ["has_prior_cba_experience_before_t", "prior_cba_seasons_before_t", "prior_cba_last_seen_gap"]
    if not any(field in row.index for field in cba_fields):
        st.caption(t("no_cba_context_field", lang))
    cba_context_fields = [
        ("has_prior_cba_experience_before_t", "has_prior_cba_experience_before_t"),
        ("prior_cba_seasons_before_t", "prior_cba_seasons_before_t"),
        ("prior_cba_last_seen_gap", "prior_cba_last_seen_gap"),
    ]
    st.table(sanitize_dataframe_for_streamlit(_detail_rows(row, cba_context_fields, bool_fields={"has_prior_cba_experience_before_t"}, lang=lang)))

    st.markdown(f"**{t('search_review_links', lang)}**")
    st.caption(t("manual_search_caption", lang))
    l1, l2, l3 = st.columns(3)
    with l1:
        _link_button(t("youtube_search", lang), _youtube_search_url(row), lang)
    with l2:
        _link_button(t("google_video_search", lang), _google_video_search_url(row), lang)
    with l3:
        _link_button(t("basketball_reference_search", lang), _basketball_reference_search_url(row), lang)

    o1, o2, o3 = st.columns(3)
    with o1:
        _link_button(t("nba_stats", lang), row.get("nba_stats_url"), lang)
    with o2:
        _link_button(t("euroleague_search", lang), row.get("euroleague_profile_or_search_url"), lang)
    with o3:
        _link_button(t("league_official_search", lang), _league_official_search_url(row), lang)


# 功能：调用统一的候选人详情面板。
def detail_panel(player: pd.Series) -> None:
    render_candidate_detail(player)

# 功能：运行当前模块的主要流程并保存输出。
def main() -> None:
    st.set_page_config(page_title="CBA Import Candidate Scouting Dashboard", layout="wide")
    language_options = {"\u4e2d\u6587": "zh", "English": "en"}
    current_lang = str(st.session_state.get("language", "zh"))
    default_lang_display = "English" if current_lang == "en" else "\u4e2d\u6587"
    selected_lang_display = st.sidebar.selectbox(
        "Language / \u8bed\u8a00",
        list(language_options.keys()),
        index=list(language_options.keys()).index(default_lang_display),
    )
    lang = language_options[selected_lang_display]
    st.session_state["language"] = lang

    st.title(t("app_title", lang))

    ensure_registry()
    sources = list_sources()
    if sources.empty:
        st.warning(t("no_source_found", lang))
        return
    display_names = sources["display_name"].astype(str).tolist()
    default_index = display_names.index(DEFAULT_DISPLAY_NAME) if DEFAULT_DISPLAY_NAME in display_names else 0
    selected_display_name = st.sidebar.selectbox(t("data_source", lang), display_names, index=default_index, format_func=_sidebar_display_name)
    selected_rows = sources[sources["display_name"].astype(str).eq(str(selected_display_name))]
    selected_source_meta = selected_rows.iloc[0].to_dict() if not selected_rows.empty else {}
    _source_management_panel(selected_source_meta, sources, lang)

    df, source_meta = load_data(selected_display_name)
    if df.empty:
        st.warning(t("source_empty", lang))
        return
    is_demo_source = str(source_meta.get("source_type", "")) == "demo_fallback"
    if is_demo_source:
        st.warning(t("demo_data_warning", lang))

    filtered = filtered_data(df, lang)
    tabs = st.tabs([
        t("tab_overview", lang),
        t("tab_candidates", lang),
        t("tab_new_data", lang),
        t("tab_highlights", lang),
        t("tab_full_data", lang),
    ])

    with tabs[0]:
        st.subheader(t("tab_overview", lang))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("candidate_count", lang), len(filtered))
        c2.metric(t("recommendation_season", lang), _display_value(_first_available_value(df, ["recommendation_season", "season"])))
        c3.metric(t("league_count", lang), filtered["league"].dropna().astype(str).nunique() if "league" in filtered.columns else 0)
        c4.metric(t("has_cba_experience", lang), safe_bool_count(filtered, "has_prior_cba_experience_before_t"))
        st.write(t("demo_method_statement" if is_demo_source else "final_method_statement", lang))
        st.info(t("scouting_disclaimer", lang))
        st.markdown(f"**{t('current_data_source', lang)}**")
        st.table(sanitize_dataframe_for_streamlit(_selected_source_metadata(source_meta)))
        with st.expander(t("available_data_sources", lang), expanded=False):
            st.dataframe(sanitize_dataframe_for_streamlit(_available_sources_table(sources)), width="stretch", hide_index=True)
        with st.expander(t("method_notes_expander", lang), expanded=False):
            st.markdown(t("method_notes", lang))

    with tabs[1]:
        st.subheader(t("tab_candidates", lang))
        candidate_table(filtered)
        if not filtered.empty:
            selected_rank = st.selectbox(t("open_candidate_detail", lang), filtered["rank"].astype(int).tolist(), format_func=lambda r: f"#{r}")
            render_candidate_detail(filtered[filtered["rank"].eq(selected_rank)].iloc[0], source_meta, lang)

    with tabs[2]:
        st.subheader(t("tab_new_data", lang))
        st.warning(t("new_data_warning", lang))
        st.markdown(t("new_data_help", lang))

        st.markdown(f"### {t('upload_local_file', lang)}")
        upload_display_name = st.text_input(t("upload_display_name", lang), value="Uploaded candidates")
        uploaded = st.file_uploader(t("upload_file_label", lang), type=["csv", "xlsx", "xls", "json"])
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix.lower()
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = UPLOAD_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{Path(uploaded.name).name}"
            saved_path.write_bytes(uploaded.getbuffer())
            try:
                if suffix == ".csv":
                    raw_preview = pd.read_csv(saved_path)
                elif suffix in {".xlsx", ".xls"}:
                    raw_preview = pd.read_excel(saved_path)
                else:
                    raw_preview = pd.read_json(saved_path)
                st.write(t("detected_columns", lang), list(raw_preview.columns))
                missing = [c for c in REQUIRED_COLUMNS if c not in {str(x).strip().lower() for x in raw_preview.columns}]
                if missing:
                    st.caption(t("missing_required", lang, cols=", ".join(missing)))
                st.dataframe(sanitize_dataframe_for_streamlit(raw_preview.head(20)), width="stretch", hide_index=True)
            except Exception as exc:
                st.error(t("preview_failed", lang, error=exc))

            c1, c2 = st.columns(2)
            include_nba = c1.checkbox(t("include_nba", lang), value=False)
            include_cba = c2.checkbox(t("include_chinese_cba", lang), value=False)
            if st.button(t("generate_uploaded", lang)):
                try:
                    display_name = upload_display_name.strip() or _default_display_name("Uploaded Data")
                    ingest_candidates(saved_path, include_nba=include_nba, include_chinese_cba=include_cba, source_name=display_name)
                    ranked = rank_candidates()
                    saved_source = save_generated_recommendation_source(
                        recommendations=ranked,
                        display_name=display_name,
                        source_type="uploaded_file",
                        raw_input_file=saved_path,
                        clean_file="data/processed/new_candidates_clean.csv",
                        parse_report_file="data/reports/new_candidate_ingestion_report.csv",
                        season=", ".join(sorted(ranked.get("season", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())),
                        league=", ".join(sorted(ranked.get("league", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())),
                        notes="dashboard local upload recommendation source",
                    )
                    st.success(t("generated_count", lang, count=len(ranked)))
                    st.success(t("saved_as_source", lang, name=display_name))
                    st.caption(t("saved_file", lang, path=saved_source["recommendation_file"]))
                    new_candidate_table(ranked)
                    st.download_button(
                        t("download_new_csv", lang),
                        ranked.to_csv(index=False).encode("utf-8"),
                        file_name="new_candidate_recommendations.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(t("new_candidate_failed", lang, error=exc))
        elif NEW_CANDIDATE_OUTPUT.exists():
            st.caption(t("showing_latest_new", lang))
            new_df = pd.read_csv(NEW_CANDIDATE_OUTPUT)
            new_candidate_table(new_df)
            st.download_button(
                t("download_latest_new_csv", lang),
                new_df.to_csv(index=False).encode("utf-8"),
                file_name="new_candidate_recommendations.csv",
                mime="text/csv",
            )

        st.divider()
        st.markdown(f"### {t('paste_raw_text', lang)}")
        st.caption(t("paste_caption", lang))
        pasted_display_name = st.text_input(t("paste_display_name", lang), value="Basketball-Reference pasted table")
        pasted_text = st.text_area(t("paste_text_label", lang), height=220)
        format_hint = st.selectbox(
            t("format_hint", lang),
            [
                "Auto detect",
                "Basketball-Reference Per 36",
                "Basketball-Reference Per Game",
                "Basketball-Reference Advanced",
                "Generic CSV text",
            ],
        )
        p1, p2, p3 = st.columns(3)
        pasted_season = p1.text_input(t("season_override", lang), value="", placeholder="2025-2026")
        pasted_league = p2.text_input(t("league_override", lang), value="", placeholder="G League")
        pasted_source = p3.text_input(t("source_name", lang), value="pasted_stats_table")
        keep_all_team_rows = st.checkbox(t("keep_all_team_rows", lang), value=False)
        if st.button(t("parse_and_generate", lang)):
            if not pasted_text.strip():
                st.warning(t("paste_before_parse", lang))
            elif not pasted_season.strip() or not pasted_league.strip():
                st.warning(t("season_league_required", lang))
            else:
                try:
                    converted, parse_report = parse_pasted_text(
                        pasted_text,
                        season=pasted_season.strip(),
                        league=pasted_league.strip(),
                        source_name=pasted_source.strip() or "pasted_stats_table",
                        format_hint=format_hint,
                        keep_all_team_rows=keep_all_team_rows,
                    )
                    st.success(t("parsed_rows", lang, count=len(converted)))
                    st.dataframe(sanitize_dataframe_for_streamlit(parse_report), width="stretch", hide_index=True)
                    st.write(t("converted_preview", lang))
                    st.dataframe(sanitize_dataframe_for_streamlit(converted.head(20)), width="stretch", hide_index=True)
                    raw_text_path = UPLOAD_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_pasted_candidate_text.txt"
                    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    raw_text_path.write_text(pasted_text, encoding="utf-8")
                    display_name = pasted_display_name.strip() or _default_display_name("Pasted Basketball-Reference")
                    ingest_candidates("data/manual/uploads/pasted_candidate_text_latest.csv", source_name=pasted_source.strip() or display_name)
                    ranked = rank_candidates()
                    saved_source = save_generated_recommendation_source(
                        recommendations=ranked,
                        display_name=display_name,
                        source_type="pasted_text",
                        raw_input_file=raw_text_path,
                        clean_file="data/processed/new_candidates_clean.csv",
                        parse_report_file="data/reports/pasted_candidate_parse_report.csv",
                        season=pasted_season.strip(),
                        league=pasted_league.strip(),
                        notes=f"pasted text recommendation source; format_hint={format_hint}",
                    )
                    st.success(t("saved_as_source", lang, name=display_name))
                    st.caption(t("saved_file", lang, path=saved_source["recommendation_file"]))
                    new_candidate_table(ranked)
                    st.download_button(
                        t("download_new_csv", lang),
                        ranked.to_csv(index=False).encode("utf-8"),
                        file_name="new_candidate_recommendations.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(t("pasted_failed", lang, error=exc))

    with tabs[3]:
        st.subheader(t("tab_highlights", lang))
        st.caption(t("highlight_caption", lang))
        if filtered.empty:
            st.write(t("no_filter_match", lang))
        else:
            names = [f"#{int(row.rank)} {row.player_name_raw}" for row in filtered.itertuples(index=False)]
            selected = st.selectbox(t("select_player", lang), names, key="highlight_player")
            selected_rank = int(selected.split(" ", 1)[0].replace("#", ""))
            player = filtered[filtered["rank"].eq(selected_rank)].iloc[0]
            render_candidate_detail(player, source_meta, lang)

            with st.expander(t("search_links_detail", lang), expanded=False):
                st.dataframe(sanitize_dataframe_for_streamlit(_search_links_table(filtered.head(100))), width="stretch", hide_index=True)

    with tabs[4]:
        st.subheader(t("tab_full_data", lang))
        visible = filtered.drop(columns=[c for c in HIDDEN_URL_COLUMNS if c in filtered.columns], errors="ignore")
        st.dataframe(sanitize_dataframe_for_streamlit(visible.fillna("-")), width="stretch", hide_index=True)
        st.download_button(
            t("export_filtered_csv", lang),
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="cba_import_candidate_recommendations.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
