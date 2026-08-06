"""
电价查询 — 混合模式：DB 缓存优先 + Web 搜索兜底

首次查询某省某月某类型电价时，自动联网搜索真实数据并写入 DB。
再次查询同一省份+月份+类型时，直接命中缓存，毫秒级返回。
越用越快。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional, TypedDict

from src.config import PRICE_TYPE_MAP
from data.db import query_price, query_latest_price, query_trend, insert_price
from src.tools.web_search import web_search
from src.model_engine import generate_json

logger = logging.getLogger(__name__)


class PriceResult(TypedDict):
    province: str
    year_month: str
    price_type: str
    price_type_name: str
    price: Optional[float]
    unit: str
    source: str
    cached: bool
    trend: list[dict]
    search_used: bool


# ── LLM 价格提取 (Web 搜索兜底时使用) ──────────────────────────


def _extract_price_with_llm(text: str, province: str, price_type: str) -> dict | None:
    """用 LLM 从网页文本中提取电价数字"""
    prompt = f"""你是一个电价数据提取专家。从以下网页文本中提取 **{province}** 的 **{PRICE_TYPE_MAP.get(price_type, price_type)}** 数据。

要求:
1. 找到明确的电价数字 (单位: 元/千瓦时 或 元/兆瓦时)
2. 如果是元/兆瓦时，除以1000转换为元/千瓦时
3. 关注最近月份的数据
4. 如果文本中没有明确电价数字，返回 null

请只输出 JSON，不要其他文字:
{{"price": 数字或null, "unit": "元/千瓦时", "year_month": "YYYY-MM或null", "confidence": "high|medium|low", "evidence": "简短摘录原文中支持该数字的句子"}}

网页文本:
---
{text[:6000]}
---"""

    try:
        content = generate_json(
            [{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        json_match = re.search(r'\{[^{}]*"price"[^{}]*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            logger.info(f"LLM 提取结果: {data}")
            return data
        data = json.loads(content)
        return data
    except Exception as e:
        logger.warning(f"LLM 提取失败: {e}")
        return None


# ── 搜索结果直接传给 Agent（不单独抽取，由 DeepSeek 在对话中直接提取）


# ── 主查询函数 ─────────────────────────────────────────────────


def query_electricity_price(province: str, year_month: str, price_type: str) -> PriceResult:
    """
    电价查询 - DB 缓存优先 + Web 搜索兜底。
    搜索结果直接返回给 DeepSeek 提取电价，不再抓网页 + LLM 提取。
    """
    price_type_name = PRICE_TYPE_MAP.get(price_type, price_type)
    search_used = False

    # 1. 查询本地数据库
    result = query_price(province, year_month, price_type)
    if result:
        trend = query_trend(province, price_type)
        logger.info(f"✅ DB命中: {province} {year_month} {price_type_name} = {result['price']}")
        return {
            "province": province, "year_month": year_month, "price_type": price_type,
            "price_type_name": price_type_name, "price": result["price"],
            "unit": result["unit"], "source": result.get("source", "本地数据库"),
            "cached": True, "trend": trend, "search_used": False,
        }

    # 2. 数据库中无此月数据，尝试查最新月
    latest = query_latest_price(province, price_type)
    if latest:
        trend = query_trend(province, price_type)
        logger.info(f"⚠️ 无 {year_month} 数据，返回最新: {latest['year_month']}")
        return {
            "province": province, "year_month": latest["year_month"], "price_type": price_type,
            "price_type_name": price_type_name, "price": latest["price"],
            "unit": latest["unit"], "source": latest.get("source", "本地数据库"),
            "cached": True, "trend": trend, "search_used": False,
        }

    # 3. DB 无数据 → Web 搜索（两次尝试：精确+宽泛）
    logger.info(f"🔍 无 {province} {price_type_name} 缓存，搜索真实数据...")
    search_used = True

    # 第一次：精确搜索
    search_results = web_search(f"{province} {price_type_name} 元/千瓦时 最新", max_results=5)

    # 第二次：如果第一次没结果，换更宽泛的关键词
    if not search_results:
        logger.info("   第一次无结果，尝试宽泛搜索...")
        search_results = web_search(f"{province} {price_type_name} 电价", max_results=5)

    if search_results:
        source_text = "\n\n".join(
            f"[{i}] {r['title']}\n{r['snippet']}\n链接: {r['url']}"
            for i, r in enumerate(search_results, 1)
        )
        logger.info(f"搜索到 {len(search_results)} 条结果")

        # 优先用 LLM 从搜索结果中提取结构化价格 → 入库缓存 → 返回真实价格
        extracted = _extract_price_with_llm(source_text, province, price_type)
        if extracted and extracted.get("price"):
            try:
                price = float(extracted["price"])
                insert_price(
                    province, year_month, price_type, price,
                    unit="元/千瓦时",
                    source=source_text[:500],
                )
                logger.info(f"✅ 提取并入库: {province} {year_month} {price_type_name} = {price} 元/千瓦时")
                return {
                    "province": province, "year_month": year_month, "price_type": price_type,
                    "price_type_name": price_type_name, "price": price,
                    "unit": "元/千瓦时", "source": "实时 Web 搜索 (已缓存)",
                    "cached": False, "trend": [], "search_used": True,
                }
            except Exception as e:
                logger.warning(f"入库失败: {e}")

        # 提取/入库失败 → 退回把搜索结果交给 Agent 提取 (price=-1 特殊标记)
        logger.info(f"LLM 未能提取价格，退回交给 Agent 提取")
        return {
            "province": province, "year_month": year_month, "price_type": price_type,
            "price_type_name": price_type_name,
            "price": -1,  # 特殊标记：表示"有搜索结果待提取"
            "unit": "元/千瓦时",
            "source": source_text,
            "cached": False, "trend": [], "search_used": True,
        }

    return _empty_result(province, year_month, price_type, price_type_name, search_used=True)


def _empty_result(province: str, year_month: str, price_type: str,
                  price_type_name: str, search_used: bool = False) -> PriceResult:
    return {
        "province": province,
        "year_month": year_month,
        "price_type": price_type,
        "price_type_name": price_type_name,
        "price": None,
        "unit": "元/千瓦时",
        "source": "",
        "cached": False,
        "trend": [],
        "search_used": search_used,
    }
