"""
电价实时查询工具 - Web Search → Fetch → LLM 提取 → SQLite 缓存
支持三种电价类型: 上网电价 / 脱硫煤电价 / 工商业电价
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from openai import OpenAI

from src.config import PRICE_TYPE_MAP, VLLM_BASE_URL, DATA_DIR
from src.tools.web_search import web_search, web_fetch

logger = logging.getLogger(__name__)

# SQLite 数据库路径 — 优先持久化到 Google Drive
DB_PATH = Path(DATA_DIR) / "electricity_cache.db"


class PriceResult(TypedDict):
    province: str
    year_month: str
    price_type: str
    price_type_name: str
    price: float | None
    unit: str
    source: str
    cached: bool
    trend: list[dict]
    search_used: bool


# ── SQLite 缓存层 ─────────────────────────────────────────────


def _get_db() -> sqlite3.Connection:
    """获取数据库连接，自动建表"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS electricity_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province VARCHAR(20) NOT NULL,
            year_month VARCHAR(7) NOT NULL,
            price_type VARCHAR(30) NOT NULL,
            price REAL NOT NULL,
            unit VARCHAR(10) DEFAULT '元/千瓦时',
            source VARCHAR(500),
            raw_context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_price_unique
        ON electricity_prices(province, year_month, price_type)
    """)
    conn.commit()
    return conn


def _cache_get(province: str, year_month: str, price_type: str) -> dict | None:
    """从缓存查询电价"""
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT price, unit, source, created_at FROM electricity_prices "
            "WHERE province=? AND year_month=? AND price_type=?",
            (province, year_month, price_type),
        ).fetchone()
        conn.close()
        if row:
            return {"price": row[0], "unit": row[1], "source": row[2], "cached_at": row[3]}
    except Exception as e:
        logger.warning(f"缓存查询失败: {e}")
    return None


def _cache_set(province: str, year_month: str, price_type: str,
               price: float, unit: str, source: str, raw_context: str = "") -> None:
    """写入缓存"""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR REPLACE INTO electricity_prices "
            "(province, year_month, price_type, price, unit, source, raw_context, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (province, year_month, price_type, price, unit, source, raw_context),
        )
        conn.commit()
        conn.close()
        logger.info(f"缓存写入: {province} {year_month} {price_type} = {price} {unit}")
    except Exception as e:
        logger.warning(f"缓存写入失败: {e}")


def _cache_get_trend(province: str, price_type: str, months: int = 6) -> list[dict]:
    """获取某省某类电价的历史趋势"""
    try:
        conn = _get_db()
        rows = conn.execute(
            "SELECT year_month, price, source FROM electricity_prices "
            "WHERE province=? AND price_type=? "
            "ORDER BY year_month DESC LIMIT ?",
            (province, price_type, months),
        ).fetchall()
        conn.close()
        return [{"year_month": r[0], "price": r[1], "source": r[2]} for r in reversed(rows)]
    except Exception as e:
        logger.warning(f"趋势查询失败: {e}")
        return []


# ── LLM 价格提取 ──────────────────────────────────────────────


def _extract_price_with_llm(text: str, province: str, price_type: str) -> dict | None:
    """
    用 vLLM 从网页文本中提取电价数字
    返回 {"price": float, "unit": str, "confidence": str} 或 None
    """
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
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
        resp = client.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        # 提取 JSON
        json_match = re.search(r'\{[^{}]*"price"[^{}]*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            logger.info(f"LLM 提取结果: {data}")
            return data
        # 尝试直接解析整个响应
        data = json.loads(content)
        return data
    except Exception as e:
        logger.warning(f"LLM 提取失败: {e}")
        return None


# ── 主查询函数 ────────────────────────────────────────────────


def query_electricity_price(province: str, year_month: str, price_type: str) -> PriceResult:
    """
    电价实时查询 - 缓存优先 + Web Search 实时获取

    Args:
        province: 省份名称 (如 "江苏", "上海")
        year_month: 查询月份 (如 "2026-08", 默认当月)
        price_type: feed_in | desulfurized_coal | commercial_industrial

    Returns:
        PriceResult { province, price, trend, source, cached, ... }
    """
    price_type_name = PRICE_TYPE_MAP.get(price_type, price_type)
    search_used = False

    # 1. 查询缓存
    cached = _cache_get(province, year_month, price_type)
    if cached:
        trend = _cache_get_trend(province, price_type)
        logger.info(f"✅ 缓存命中: {province} {year_month} {price_type_name} = {cached['price']}")
        return {
            "province": province,
            "year_month": year_month,
            "price_type": price_type,
            "price_type_name": price_type_name,
            "price": cached["price"],
            "unit": cached["unit"],
            "source": cached["source"],
            "cached": True,
            "trend": trend,
            "search_used": False,
        }

    # 2. 缓存未命中，实时搜索
    logger.info(f"🔍 缓存未命中，实时搜索: {province} {year_month} {price_type_name}")
    search_used = True

    # 构建搜索词
    if year_month:
        query = f"{province} {price_type_name} {year_month} 元/千瓦时"
    else:
        query = f"{province} {price_type_name} 最新电价 元/千瓦时"

    # 2a. DuckDuckGo 搜索
    results = web_search(query, max_results=5)
    if not results:
        logger.warning(f"搜索无结果: {query}")
        return _empty_result(province, year_month, price_type, price_type_name, search_used=True)

    # 2b. 抓取前2个结果
    extracted_price = None
    extracted_source = ""
    extracted_context = ""

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

        # 2c. LLM 提取价格
        llm_result = _extract_price_with_llm(page_text, province, price_type)
        if llm_result and llm_result.get("price") is not None:
            extracted_price = float(llm_result["price"])
            extracted_source = f"{r.get('title', '')} - {url}"
            extracted_context = llm_result.get("evidence", "")
            confidence = llm_result.get("confidence", "medium")
            logger.info(f"提取成功: {extracted_price} 元/千瓦时 (confidence={confidence})")
            if confidence == "high":
                break  # 高置信度直接使用
        time.sleep(0.3)  # 请求间隔

    # 3. 写入缓存
    if extracted_price is not None:
        _cache_set(province, year_month, price_type, extracted_price,
                   "元/千瓦时", extracted_source, extracted_context)
        trend = _cache_get_trend(province, price_type)
        return {
            "province": province,
            "year_month": year_month,
            "price_type": price_type,
            "price_type_name": price_type_name,
            "price": extracted_price,
            "unit": "元/千瓦时",
            "source": extracted_source,
            "cached": False,
            "trend": trend,
            "search_used": True,
        }

    # 4. 未提取到价格，返回搜索结果摘要
    logger.warning(f"未能提取到 {province} {price_type_name} 的价格")
    return _empty_result(province, year_month, price_type, price_type_name, search_used=True)


def _empty_result(province: str, year_month: str, price_type: str,
                  price_type_name: str, search_used: bool = False) -> PriceResult:
    """构建空结果"""
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
