#!/usr/bin/env python3
"""
电价 Mock 数据填充脚本
生成 31 个省级行政区 × 20 个月 × 3 种电价类型的模拟数据

电价基准参考各省燃煤标杆上网电价（脱硫煤电价），
上网电价基本等同脱硫煤标杆价，工商业电价在此基础上上浮。

数据来源参考:
- 国家发改委各省燃煤发电上网电价
- 各省发改委工商业电价目录
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import get_db, insert_price, get_db_stats
from src.config import MAINLAND_PROVINCES, PRICE_TYPE_MAP, logger

# ── 各省脱硫煤标杆上网电价基准 (元/千瓦时) ──────────────────
# 基于公开的各省燃煤发电标杆上网电价数据

PROVINCE_BASE_PRICE: dict[str, float] = {
    "北京": 0.3598,
    "天津": 0.3655,
    "河北": 0.3644,
    "山西": 0.3320,
    "内蒙古": 0.2829,
    "辽宁": 0.3749,
    "吉林": 0.3731,
    "黑龙江": 0.3740,
    "上海": 0.4155,
    "江苏": 0.3910,
    "浙江": 0.4153,
    "安徽": 0.3844,
    "福建": 0.3932,
    "江西": 0.4143,
    "山东": 0.3949,
    "河南": 0.3779,
    "湖北": 0.4161,
    "湖南": 0.4500,
    "广东": 0.4530,
    "广西": 0.4207,
    "海南": 0.4298,
    "重庆": 0.3964,
    "四川": 0.4012,
    "贵州": 0.3515,
    "云南": 0.3358,
    "西藏": 0.4498,
    "陕西": 0.3545,
    "甘肃": 0.3078,
    "青海": 0.3247,
    "宁夏": 0.2595,
    "新疆": 0.2500,
}

# 工商业电价相对于脱硫煤电价的浮动比例
# 一般工商业电价 ≈ 脱硫煤电价 + 输配电价 + 政府基金
# 各地差异较大，大约在 1.5-2.5 倍之间
COMMERCIAL_MULTIPLIER: dict[str, float] = {
    "北京": 2.1, "天津": 2.0, "河北": 1.9, "山西": 1.8,
    "内蒙古": 1.7, "辽宁": 2.0, "吉林": 1.9, "黑龙江": 1.9,
    "上海": 2.2, "江苏": 2.1, "浙江": 2.2, "安徽": 1.9,
    "福建": 2.0, "江西": 1.9, "山东": 2.0, "河南": 1.9,
    "湖北": 2.0, "湖南": 2.0, "广东": 2.3, "广西": 1.8,
    "海南": 2.1, "重庆": 2.0, "四川": 1.9, "贵州": 1.8,
    "云南": 1.7, "西藏": 1.6, "陕西": 1.9, "甘肃": 1.7,
    "青海": 1.7, "宁夏": 1.6, "新疆": 1.5,
}

# 上网电价基本等于脱硫煤标杆价（风电/光伏略有差异，此处简化为等同）
FEED_IN_MULTIPLIER = 1.0


def generate_monthly_prices(base: float, multiplier: float, months: int = 20) -> list[float]:
    """
    生成月度电价序列，带合理波动

    电价通常按月微调 (±2%)，偶尔有政策性调整 (±5%)
    """
    prices = []
    current = base * multiplier

    for i in range(months):
        # 每月微小波动：以 ±1.5% 随机浮动
        monthly_change = random.uniform(-0.015, 0.015)

        # 大约每 6 个月有一次政策调整机会
        if i > 0 and i % 6 == 0:
            policy_adjust = random.uniform(-0.03, 0.03)
            current *= (1 + policy_adjust)

        current *= (1 + monthly_change)
        # 保持在合理范围内
        current = max(base * multiplier * 0.9, min(base * multiplier * 1.1, current))
        prices.append(round(current, 4))

    return prices


def generate_months(start_year: int = 2025, start_month: int = 1, count: int = 20) -> list[str]:
    """生成连续的月份列表"""
    months = []
    year, month = start_year, start_month
    for _ in range(count):
        months.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def seed_database():
    """填充 Mock 数据"""
    random.seed(42)  # 固定随机种子，保证数据可复现
    months = generate_months(2025, 1, 20)  # 2025-01 ~ 2026-08

    # 先清空已有数据
    conn = get_db()
    conn.execute("DELETE FROM electricity_prices")
    conn.commit()
    conn.close()

    total = 0
    price_types = ["feed_in", "desulfurized_coal", "commercial_industrial"]

    for province in MAINLAND_PROVINCES:
        base = PROVINCE_BASE_PRICE.get(province, 0.40)

        for pt in price_types:
            if pt == "feed_in":
                multiplier = FEED_IN_MULTIPLIER
            elif pt == "desulfurized_coal":
                multiplier = 1.0
            else:
                multiplier = COMMERCIAL_MULTIPLIER.get(province, 2.0)

            price_series = generate_monthly_prices(base, multiplier, len(months))

            for month, price in zip(months, price_series):
                insert_price(
                    province=province,
                    year_month=month,
                    price_type=pt,
                    price=price,
                    unit="元/千瓦时",
                    source=f"{province}发改委 / 电力交易中心 (Mock)",
                )
                total += 1

    logger.info(f"✅ Mock 数据填充完成: {total} 条记录")
    stats = get_db_stats()
    logger.info(f"   数据库统计: {stats}")
    return stats


if __name__ == "__main__":
    seed_database()
