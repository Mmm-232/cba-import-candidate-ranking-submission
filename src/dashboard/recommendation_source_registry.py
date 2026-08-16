"""
Module: recommendation_source_registry.py
Purpose: Maintain registry metadata for default/user/parsed/API recommendation sources and their lifecycle.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from .build_role_aware_scores import add_role_aware_scores


REPORTS_DIR = Path("data/reports")
REGISTRY_PATH = REPORTS_DIR / "recommendation_sources_registry.csv"
USER_RECOMMENDATION_DIR = REPORTS_DIR / "user_recommendations"
ENRICHED_DEFAULT_PATH = REPORTS_DIR / "frontend_recommendations_enriched.csv"
BASE_DEFAULT_PATH = REPORTS_DIR / "frontend_recommendations.csv"
DEMO_DEFAULT_PATH = Path("data/demo/frontend_recommendations_demo.csv")
DEMO_SOURCE_ID = "demo_fallback"
DEMO_DISPLAY_NAME = "演示数据 / Demo only（非真实模型结果）"

REGISTRY_COLUMNS = [
    "source_id",
    "display_name",
    "source_type",
    "recommendation_file",
    "raw_input_file",
    "clean_file",
    "parse_report_file",
    "created_at",
    "season",
    "league",
    "row_count",
    "notes",
]

DEFAULT_SOURCE_ID = "final_2024_2025_rule_based"
DEFAULT_DISPLAY_NAME = "\u9ed8\u8ba4\u63a8\u8350\u540d\u5355\uff1a2024-2025"


# 功能：清理数据源登记表中的显示名称。
def _clean_registry_display_names(registry: pd.DataFrame) -> pd.DataFrame:
    registry = registry.copy()
    if "display_name" not in registry.columns:
        return registry
    default_mask = registry.get("source_id", pd.Series(index=registry.index, dtype=object)).astype(str).eq(DEFAULT_SOURCE_ID)
    registry.loc[default_mask, "display_name"] = DEFAULT_DISPLAY_NAME

    # 功能：为缺失字段提供兼容的默认值。
    def fallback(row: pd.Series) -> str:
        value = str(row.get("display_name", "") or "").strip()
        if not value or "????" in value:
            return str(row.get("source_id", "recommendation_source"))
        return value

    registry["display_name"] = registry.apply(fallback, axis=1)
    return registry




USER_SOURCE_TYPES = {"uploaded_file", "pasted_text", "api_import"}


# 功能：判断数据源是否由用户上传、粘贴或 API 导入。
def is_user_source(row: pd.Series | dict[str, object]) -> bool:
    return str(row.get("source_type", "")).strip() in USER_SOURCE_TYPES


# 功能：确认文件位于允许删除的用户推荐目录中。
def is_safe_user_recommendation_path(path: str | Path | None) -> bool:
    if not path:
        return False
    try:
        candidate = Path(path).resolve()
        base = USER_RECOMMENDATION_DIR.resolve()
        return candidate == base or base in candidate.parents
    except (OSError, RuntimeError, ValueError):
        return False

# 功能：生成当前时间标记。
def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# 功能：把数据源名称转换成安全且稳定的标识符。
def safe_source_id(display_name: str) -> str:
    text = str(display_name or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "recommendation_source"


# 功能：选择正式推荐文件；缺失时返回演示数据。
def _default_recommendation_file() -> Path:
    return ENRICHED_DEFAULT_PATH if ENRICHED_DEFAULT_PATH.exists() else BASE_DEFAULT_PATH


# 功能：为推荐表补齐人工搜索和核查链接。
def ensure_search_links(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "player_name_raw" not in out.columns:
        return out
    if "league" not in out.columns:
        out["league"] = ""

    query = (
        out["player_name_raw"].fillna("").astype(str).str.strip()
        + " "
        + out["league"].fillna("").astype(str).str.strip()
        + " basketball highlights"
    ).str.strip()
    query = query.where(query.str.len().gt(0), out["player_name_raw"].fillna("").astype(str) + " basketball highlights")

    if "video_search_query" not in out.columns:
        out["video_search_query"] = query
    else:
        out["video_search_query"] = out["video_search_query"].fillna("")
        out.loc[out["video_search_query"].astype(str).str.strip().eq(""), "video_search_query"] = query

    youtube = "https://www.youtube.com/results?search_query=" + out["video_search_query"].fillna("").astype(str).map(quote_plus)
    google_video = "https://www.google.com/search?tbm=vid&q=" + out["video_search_query"].fillna("").astype(str).map(quote_plus)
    bref_query = out["player_name_raw"].fillna("").astype(str).map(lambda name: f'site:basketball-reference.com "{name}" basketball')
    bref = "https://www.google.com/search?q=" + bref_query.map(quote_plus)
    official_query = out.apply(
        lambda row: f'"{row.get("player_name_raw", "")}" "{row.get("league", "")}" official basketball stats',
        axis=1,
    )
    official = "https://www.google.com/search?q=" + official_query.map(quote_plus)

    for col, values in {
        "youtube_search_url": youtube,
        "google_video_search_url": google_video,
        "basketball_reference_search_url": bref,
        "league_official_search_url": official,
    }.items():
        if col not in out.columns:
            out[col] = values
        else:
            out[col] = out[col].fillna("")
            out.loc[out[col].astype(str).str.strip().eq(""), col] = values
    return out


# 功能：统一不同推荐来源的字段和显示格式。
def standardize_recommendation_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_search_links(df)
    if "rank" not in out.columns and "new_rank" in out.columns:
        out["rank"] = out["new_rank"]
    if "score" not in out.columns and "recommendation_score" in out.columns:
        out["score"] = out["recommendation_score"]
    if "recommendation_season" not in out.columns and "season" in out.columns:
        out["recommendation_season"] = out["season"]
    if "source" not in out.columns:
        out["source"] = pd.NA
    if "rank" in out.columns:
        out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
        out = out.sort_values("rank", na_position="last")
    return out


# 功能：创建或修复本地推荐数据源登记表。
def ensure_registry() -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    USER_RECOMMENDATION_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH, encoding="utf-8-sig")
    else:
        registry = pd.DataFrame(columns=REGISTRY_COLUMNS)
    for col in REGISTRY_COLUMNS:
        if col not in registry.columns:
            registry[col] = pd.NA
    registry = _clean_registry_display_names(registry)

    default_file = _default_recommendation_file()
    if default_file.exists():
        registry = registry[~registry["source_id"].astype(str).eq(DEMO_SOURCE_ID)]
        if not registry["source_id"].astype(str).eq(DEFAULT_SOURCE_ID).any():
            row_count = len(pd.read_csv(default_file, encoding="utf-8-sig"))
            default_row = {
                "source_id": DEFAULT_SOURCE_ID,
                "display_name": DEFAULT_DISPLAY_NAME,
                "source_type": "final_default",
                "recommendation_file": str(default_file),
                "raw_input_file": "",
                "clean_file": "",
                "parse_report_file": "",
                "created_at": datetime.fromtimestamp(default_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                "season": "2024-2025",
                "league": "",
                "row_count": row_count,
                "notes": "final common CBA source league pool + rule-based baseline",
            }
            registry = pd.concat([pd.DataFrame([default_row]), registry], ignore_index=True)
    elif DEMO_DEFAULT_PATH.exists():
        registry = registry[~registry["source_id"].astype(str).eq(DEFAULT_SOURCE_ID)]
        if not registry["source_id"].astype(str).eq(DEMO_SOURCE_ID).any():
            demo_row = {
                "source_id": DEMO_SOURCE_ID,
                "display_name": DEMO_DISPLAY_NAME,
                "source_type": "demo_fallback",
                "recommendation_file": str(DEMO_DEFAULT_PATH),
                "raw_input_file": "",
                "clean_file": "",
                "parse_report_file": "",
                "created_at": datetime.fromtimestamp(DEMO_DEFAULT_PATH.stat().st_mtime, tz=timezone.utc).isoformat(),
                "season": "DEMO-ONLY",
                "league": "Fictional demo leagues",
                "row_count": len(pd.read_csv(DEMO_DEFAULT_PATH, encoding="utf-8-sig")),
                "notes": "fictional interface demo; not historical model output or a real recommendation",
            }
            registry = pd.concat([pd.DataFrame([demo_row]), registry], ignore_index=True)
    registry = registry[REGISTRY_COLUMNS].drop_duplicates("source_id", keep="last")
    registry.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    return registry


# 功能：列出当前可以在 Dashboard 中选择的数据源。
def list_sources() -> pd.DataFrame:
    registry = ensure_registry()
    if registry.empty:
        return registry
    order = registry["source_type"].astype(str).ne("final_default").astype(int)
    return registry.assign(_order=order).sort_values(["_order", "created_at"], ascending=[True, False]).drop(columns="_order")


# 功能：按界面显示名称查找数据源登记记录。
def get_source_by_display_name(display_name: str) -> dict[str, object] | None:
    registry = list_sources()
    matches = registry[registry["display_name"].astype(str).eq(str(display_name))]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


# 功能：读取用户当前选择的数据源推荐表。
def load_selected_recommendations(display_name: str | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    registry = list_sources()
    if registry.empty:
        return pd.DataFrame(), {}
    if display_name:
        source = get_source_by_display_name(display_name)
    else:
        source = None
    if source is None:
        source = registry.iloc[0].to_dict()
    path = Path(str(source.get("recommendation_file", "")))
    if not path.exists():
        return pd.DataFrame(), source
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = standardize_recommendation_frame(df)
    df["dashboard_data_file"] = str(path)
    df["dashboard_source_id"] = source.get("source_id", "")
    df["dashboard_display_name"] = source.get("display_name", "")
    return df, source


# 功能：在源文件存在时安全复制到用户推荐目录。
def _copy_if_exists(path: str | Path | None, target: Path) -> str:
    if not path:
        return ""
    source = Path(path)
    if not source.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


# 功能：把一个推荐数据源登记到本地 registry。
def register_source(
    *,
    source_id: str,
    display_name: str,
    source_type: str,
    recommendation_file: str | Path,
    raw_input_file: str | Path | None = None,
    clean_file: str | Path | None = None,
    parse_report_file: str | Path | None = None,
    season: str | None = None,
    league: str | None = None,
    row_count: int | None = None,
    notes: str | None = None,
) -> pd.DataFrame:
    registry = ensure_registry()
    row = {
        "source_id": source_id,
        "display_name": display_name,
        "source_type": source_type,
        "recommendation_file": str(recommendation_file),
        "raw_input_file": str(raw_input_file or ""),
        "clean_file": str(clean_file or ""),
        "parse_report_file": str(parse_report_file or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season": season or "",
        "league": league or "",
        "row_count": row_count if row_count is not None else "",
        "notes": notes or "",
    }
    registry = registry[~registry["source_id"].astype(str).eq(source_id)]
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry = registry[REGISTRY_COLUMNS].drop_duplicates("source_id", keep="last")
    registry.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    return registry


# 功能：保存新生成的推荐结果并登记为可选数据源。
def save_generated_recommendation_source(
    *,
    recommendations: pd.DataFrame,
    display_name: str,
    source_type: str,
    raw_input_file: str | Path | None = None,
    clean_file: str | Path | None = None,
    parse_report_file: str | Path | None = None,
    season: str | None = None,
    league: str | None = None,
    notes: str | None = None,
) -> dict[str, str]:
    ensure_registry()
    source_id = f"{safe_source_id(display_name)}_{_now_stamp()}"
    output_dir = USER_RECOMMENDATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation_path = output_dir / f"{source_id}_recommendations.csv"
    clean_path = output_dir / f"{source_id}_clean.csv"
    parse_report_path = output_dir / f"{source_id}_parse_report.csv"

    raw_copy = ""
    if raw_input_file:
        raw_source = Path(raw_input_file)
        raw_copy = _copy_if_exists(raw_source, output_dir / f"{source_id}_raw_input{raw_source.suffix or '.txt'}")

    clean_copy = _copy_if_exists(clean_file, clean_path)
    parse_copy = _copy_if_exists(parse_report_file, parse_report_path)

    out = standardize_recommendation_frame(recommendations)
    out = add_role_aware_scores(out, write_reports=False, source_label=display_name)
    out.to_csv(recommendation_path, index=False, encoding="utf-8-sig")

    register_source(
        source_id=source_id,
        display_name=display_name,
        source_type=source_type,
        recommendation_file=recommendation_path,
        raw_input_file=raw_copy,
        clean_file=clean_copy,
        parse_report_file=parse_copy,
        season=season,
        league=league,
        row_count=len(out),
        notes=notes,
    )
    return {
        "source_id": source_id,
        "recommendation_file": str(recommendation_path),
        "raw_input_file": raw_copy,
        "clean_file": clean_copy,
        "parse_report_file": parse_copy,
    }


# 功能：安全删除用户数据源登记及允许范围内的关联文件。
def delete_source(source_id: str, delete_files: bool = False) -> dict[str, object]:
    registry = ensure_registry()
    source_id = str(source_id or "").strip()
    result: dict[str, object] = {
        "deleted_source_id": source_id,
        "deleted_display_name": "",
        "registry_row_removed": False,
        "files_deleted": [],
        "files_skipped": [],
        "error": "",
    }
    if not source_id:
        result["error"] = "missing source_id"
        return result

    matches = registry[registry["source_id"].astype(str).eq(source_id)]
    if matches.empty:
        result["error"] = f"source_id not found: {source_id}"
        return result

    row = matches.iloc[0]
    result["deleted_display_name"] = str(row.get("display_name", ""))
    source_type = str(row.get("source_type", ""))
    if source_type == "final_default" or not is_user_source(row):
        result["error"] = "default or non-user source cannot be deleted"
        return result

    file_columns = ["recommendation_file", "raw_input_file", "clean_file", "parse_report_file"]
    files_to_consider = [row.get(col, "") for col in file_columns]

    registry = registry[~registry["source_id"].astype(str).eq(source_id)]
    registry = registry[REGISTRY_COLUMNS].drop_duplicates("source_id", keep="last")
    registry.to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")
    result["registry_row_removed"] = True

    if delete_files:
        for file_value in files_to_consider:
            file_text = str(file_value or "").strip()
            if not file_text or file_text.lower() in {"nan", "none"}:
                continue
            path = Path(file_text)
            if not is_safe_user_recommendation_path(path):
                result["files_skipped"].append(file_text)
                continue
            if path.exists() and path.is_file():
                path.unlink()
                result["files_deleted"].append(str(path))
            else:
                result["files_skipped"].append(file_text)
    else:
        result["files_skipped"] = [
            str(x) for x in files_to_consider if str(x or "").strip() and str(x).strip().lower() not in {"nan", "none"}
        ]
    return result



# 功能：兼容旧调用名称并转交给安全删除函数。
def delete_recommendation_source(*args, **kwargs) -> dict[str, object]:
    """Backward-compatible alias for older dashboard code paths."""
    return delete_source(*args, **kwargs)


# 功能：兼容旧调用名称并确保 registry 可用。
def ensure_recommendation_source_registry() -> pd.DataFrame:
    """Backward-compatible alias for registry initialisation."""
    return ensure_registry()


# 功能：兼容旧调用名称并读取全部数据源。
def load_recommendation_sources() -> pd.DataFrame:
    """Backward-compatible alias for listing registered recommendation sources."""
    return list_sources()
