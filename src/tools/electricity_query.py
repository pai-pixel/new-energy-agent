"""
电价查询工具 - 本地数据库优先 + Web Search 兜底
支持三种电价类型: 上网电价 / 脱硫煤电价 / 工商业电价
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional, TypedDict

from src.config import PRICE_TYPE_MAP
from data.db import query_price, query_latest_price, query_trend, insert_price
from src.tools.web_search import web_search, web_fetch
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


# ── Web 搜索兜底 ───────────────────────────────────────────────


def _search_and_extract(province: str, year_month: str, price_type: str) -> tuple[float | None, str]:
    """Web 搜索 + LLM 提取电价，返回 (price, source)"""
    price_type_name = PRICE_TYPE_MAP.get(price_type, price_type)
    query = f"{province} {price_type_name} {year_month} 元/千瓦时"
    results = web_search(query, max_results=5)

    if not results:
        return None, ""

    for r in results[:3]:
        url = r.get("url", "")
        if not url:
            continue
        try:
            page_text = web_fetch(url, timeout=12)
            if not page_text or len(page_text) < 100:
                continue
        except Exception as e:
            logger.info(f"跳过 {url[:50]}...: {e}")
            continue

        llm_result = _extract_price_with_llm(page_text, province, price_type)
        if llm_result and llm_result.get("price") is not None:
            price = float(llm_result["price"])
            source = f"{r.get('title', '')} - {url}"
            confidence = llm_result.get("confidence", "medium")
            logger.info(f"Web 提取成功: {price} 元/千瓦时 (confidence={confidence})")
            return price, source
        time.sleep(0.3)

    return None, ""


# ── 主查询函数 ─────────────────────────────────────────────────


def query_electricity_price(province: str, year_month: str, price_type: str) -> PriceResult:
    """
    电价查询 - 本地数据库优先，Web 搜索兜底

    Args:
        province: 省份名称 (如 "江苏", "上海")
        year_month: 查询月份 (如 "2026-08", 默认当月)
        price_type: feed_in | desulfurized_coal | commercial_industrial

    Returns:
        PriceResult
    """
    price_type_name = PRICE_TYPE_MAP.get(price_type, price_type)
    search_used = False

    # 1. 查询本地数据库
    result = query_price(province, year_month, price_type)

    if result:
        trend = query_trend(province, price_type)
        logger.info(f"✅ 数据库命中: {province} {year_month} {price_type_name} = {result['price']}")
        return {
            "province": province,
            "year_month": year_month,
            "price_type": price_type,
            "price_type_name": price_type_name,
            "price": result["price"],
            "unit": result["unit"],
            "source": result.get("source", "本地数据库"),
            "cached": True,
            "trend": trend,
            "search_used": False,
        }

    # 2. 数据库中无此月数据，尝试查最新月
    latest = query_latest_price(province, price_type)
    if latest:
        trend = query_trend(province, price_type)
        logger.info(f"⚠️ 无 {year_month} 数据，返回最新: {latest['year_month']}")
        return {
            "province": province,
            "year_month": latest["year_month"],
            "price_type": price_type,
            "price_type_name": price_type_name,
            "price": latest["price"],
            "unit": latest["unit"],
            "source": latest.get("source", "本地数据库"),
            "cached": True,
            "trend": trend,
            "search_used": False,
        }

    # 3. 本地数据库完全无数据，Web 搜索兜底
    logger.info(f"🔍 数据库中无 {province} {price_type_name} 数据，Web 搜索兜底")
    search_used = True
    web_price, web_source = _search_and_extract(province, year_month, price_type)

    if web_price is not None:
        # 写入数据库以便下次使用
        insert_price(province, year_month, price_type, web_price,
                     "元/千瓦时", web_source)
        trend = query_trend(province, price_type)
        return {
            "province": province,
            "year_month": year_month,
            "price_type": price_type,
            "price_type_name": price_type_name,
            "price": web_price,
            "unit": "元/千瓦时",
            "source": web_source,
            "cached": False,
            "trend": trend,
            "search_used": True,
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
