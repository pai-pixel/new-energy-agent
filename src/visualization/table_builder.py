"""
表格构建器 - 电价查询结果 DataFrame 格式化
"""

from __future__ import annotations

import pandas as pd

from src.config import PRICE_TYPE_MAP


def build_price_table(records: list[dict], price_type: str | None = None) -> pd.DataFrame | None:
    """
    将电价记录列表转为格式化 DataFrame

    Args:
        records: [{"year_month": "2026-03", "price": 0.415, "source": "..."}, ...]
        price_type: 可选的类型过滤

    Returns:
        pandas DataFrame 或 None
    """
    if not records:
        return None

    df = pd.DataFrame(records)

    # 重命名列
    column_map = {
        "province": "省份",
        "year_month": "月份",
        "price": "电价 (元/kWh)",
        "price_type": "电价类型",
        "source": "数据来源",
    }

    # 如果 records 中有 price_type_name，用它
    if "price_type_name" in df.columns:
        column_map["price_type_name"] = "电价类型"

    df_display = df.rename(columns=column_map)

    # 只保留关键列
    display_cols = [c for c in column_map.values() if c in df_display.columns]
    df_display = df_display[display_cols]

    # 格式化
    if "电价 (元/kWh)" in df_display.columns:
        df_display["电价 (元/kWh)"] = df_display["电价 (元/kWh)"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

    # 电价类型映射中文
    if "电价类型" in df_display.columns:
        df_display["电价类型"] = df_display["电价类型"].map(PRICE_TYPE_MAP).fillna(df_display["电价类型"])

    return df_display


def build_cache_status_table(stats: dict) -> pd.DataFrame:
    """构建缓存状态统计表"""
    rows = []
    for key, count in stats.items():
        rows.append({"项目": key, "数量": count})
    return pd.DataFrame(rows) if rows else None
