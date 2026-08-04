"""
Plotly 图表构建器 - 电价趋势折线图 + 多省对比柱状图
配色: 绿色系 (新能源主题)
"""

import logging
from typing import Any

import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# ── 新能源配色方案 ────────────────────────────────────────────

COLORS = {
    "primary": "#16a34a",      # 主绿
    "secondary": "#22c55e",    # 亮绿
    "tertiary": "#86efac",     # 浅绿
    "accent": "#15803d",       # 深绿
    "grid": "#e5e7eb",         # 网格线
    "bg": "#ffffff",           # 背景
    "text": "#1f2937",         # 文字
    "feed_in": "#16a34a",      # 上网电价 - 绿
    "desulfurized_coal": "#f59e0b",  # 脱硫煤 - 琥珀
    "commercial_industrial": "#3b82f6",  # 工商业 - 蓝
}

PRICE_TYPE_COLORS = {
    "feed_in": COLORS["feed_in"],
    "desulfurized_coal": COLORS["desulfurized_coal"],
    "commercial_industrial": COLORS["commercial_industrial"],
}

PRICE_TYPE_LABELS = {
    "feed_in": "上网电价",
    "desulfurized_coal": "脱硫煤电价",
    "commercial_industrial": "工商业电价",
}


def build_trend_chart(records: list[dict], province: str, price_type: str) -> go.Figure | None:
    """
    构建单省电价趋势折线图

    Args:
        records: [{"year_month": "2026-03", "price": 0.415}, ...]
        province: 省份名
        price_type: 电价类型

    Returns:
        Plotly Figure 或 None
    """
    if not records or len(records) < 2:
        return None

    months = [r["year_month"] for r in records]
    prices = [r["price"] for r in records]
    color = PRICE_TYPE_COLORS.get(price_type, COLORS["primary"])
    label = PRICE_TYPE_LABELS.get(price_type, price_type)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months,
        y=prices,
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2.5),
        marker=dict(color=color, size=8, line=dict(color="white", width=1)),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(color)}, 0.1)",
        hovertemplate="%{x}<br>%{y:.4f} 元/千瓦时<extra></extra>",
    ))

    fig.update_layout(
        title=f"📈 {province} {label} 趋势 ({months[0]} ~ {months[-1]})",
        xaxis_title=None,
        yaxis_title="元/千瓦时",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", color=COLORS["text"]),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )

    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], tickformat="%Y-%m")
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], tickformat=".3f")

    return fig


def build_comparison_chart(records_by_province: dict[str, list[dict]],
                           price_type: str) -> go.Figure | None:
    """
    构建多省同类型电价对比柱状图

    Args:
        records_by_province: {"江苏": [records], "浙江": [records]}
        price_type: 电价类型

    Returns:
        Plotly Figure 或 None
    """
    if not records_by_province:
        return None

    color = PRICE_TYPE_COLORS.get(price_type, COLORS["primary"])
    label = PRICE_TYPE_LABELS.get(price_type, price_type)

    # 取每个省最新的一个月数据
    province_names = []
    prices = []
    months = []
    for prov, records in records_by_province.items():
        if records:
            latest = records[-1]  # 已按时间升序排列
            province_names.append(prov)
            prices.append(latest["price"])
            months.append(latest["year_month"])

    if not province_names:
        return None

    fig = go.Figure()

    # 柱状图
    fig.add_trace(go.Bar(
        x=province_names,
        y=prices,
        name=label,
        marker=dict(
            color=[COLORS["primary"]] * len(province_names),
            line=dict(color=COLORS["accent"], width=1),
        ),
        text=[f"{p:.4f}" for p in prices],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.4f} 元/千瓦时 (%{customdata})<extra></extra>",
        customdata=months,
    ))

    fig.update_layout(
        title=f"📊 各省 {label} 对比",
        xaxis_title=None,
        yaxis_title="元/千瓦时",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Inter, PingFang SC, Microsoft YaHei, sans-serif", color=COLORS["text"]),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], tickformat=".3f")

    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """#16a34a → "22, 163, 74" """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"
