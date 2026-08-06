"""
电价查询 — 混合模式：DB 缓存优先 + Web 搜索兜底

首次查询某省某月某类型电价时，自动联网搜索真实数据并写入 DB。
再次查询同一省份+月份+类型时，直接命中缓存，毫秒级返回。
越用越快。

流程:
1. 查本地 SQLite (query_price) → 命中即返回
2. 无该月数据 → 查最新月 (query_latest_price)
3. 仍无 → Web 搜索真实数据 → LLM 提取价格 → insert_price 入库 → 返回
   (提取失败则返回 price=-1, 交由对话层的 DeepSeek 兜底提取)
"""

from __future__ import annotations

import json                                    # 解析 LLM 返回的 JSON(提取价格)
import logging                                 # 模块 logger
import re                                      # 从 LLM 文本中抽取 JSON 片段
import time                                    # 记录搜索/提取耗时
from datetime import datetime                  # 默认月份
from typing import Optional, TypedDict         # 类型标注

from src.config import PRICE_TYPE_MAP          # 电价类型英文key→中文名
from data.db import query_price, query_latest_price, query_trend, insert_price  # 数据库操作
from src.tools.web_search import web_search    # Web搜索工具
from src.model_engine import generate_json     # LLM 结构化输出(提取价格)

# 模块级 logger
logger = logging.getLogger(__name__)


class PriceResult(TypedDict):
    """电价查询的返回结构(类型标注)"""
    province: str                              # 省份(已去后缀)
    year_month: str                            # 月份 YYYY-MM
    price_type: str                            # 类型英文key
    price_type_name: str                       # 类型中文名
    price: float | None                        # 价格; -1=有搜索待提取, None=无数据
    unit: str                                  # 单位
    source: str                                # 数据来源
    cached: bool                               # 是否命中缓存
    trend: list                                # 历史趋势
    search_used: bool                          # 是否使用了Web搜索


def _extract_price_with_llm(text: str, province: str, price_type: str) -> dict | None:
    """
    用 LLM 从搜索结果文本中提取电价数字, 返回结构化 dict。
    这是"搜索→入库"闭环的关键一环:
    把非结构化的网页摘要交给 DeepSeek, 让它抽出 {price, unit, year_month, ...}。
    提取失败返回 None(上层回退到对话层兜底)。
    """
    # 构造提取提示词: 明确要求输出 JSON, 单位统一为元/千瓦时
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

    t0 = time.time()                              # 计时
    try:
        # 低温度调用, 保证 JSON 输出稳定
        content = generate_json(
            [{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        # 先用正则抽取最内层含 "price" 的 JSON 对象(模型偶发夹杂多余文字)
        json_match = re.search(r'\{[^{}]*"price"[^{}]*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            logger.info(f"LLM 提取结果: {data} (耗时{time.time()-t0:.1f}s)")
            return data
        data = json.loads(content)                # 正则没抓到则整体解析
        return data
    except Exception as e:
        logger.warning(f"LLM 提取失败: {e}")      # 失败不中断, 走兜底路径
        return None


# ── 主查询函数 ─────────────────────────────────────────────────


def query_electricity_price(province: str, year_month: str, price_type: str) -> PriceResult:
    """
    电价查询 - DB 缓存优先 + Web 搜索兜底。
    搜索结果直接返回给 DeepSeek 提取电价，不再抓网页 + LLM 提取。
    返回 PriceResult: price>0=真实价格, -1=有搜索待提取, None=无数据。
    """
    price_type_name = PRICE_TYPE_MAP.get(price_type, price_type)  # 英文key→中文
    search_used = False                           # 标记是否走了Web搜索

    # 1. 查询本地数据库(精确匹配 省份+月份+类型)
    result = query_price(province, year_month, price_type)
    if result:
        # 命中缓存: 附带最近12月趋势
        trend = query_trend(province, price_type)
        logger.info(f"✅ DB缓存命中: {province} {year_month} {price_type_name} = {result['price']}")
        return {
            "province": province, "year_month": year_month, "price_type": price_type,
            "price_type_name": price_type_name, "price": result["price"],
            "unit": result["unit"], "source": result.get("source", "本地数据库"),
            "cached": True, "trend": trend, "search_used": False,
        }

    # 2. 数据库中无此月数据, 尝试返回最新月(降级但仍有数据)
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

    # 3. DB 无数据 → Web 搜索真实数据(两次尝试: 精确+宽泛)
    logger.info(f"🔍 无 {province} {price_type_name} 缓存，搜索真实数据...")
    search_used = True

    # 第一次: 精确搜索(带单位+时间词, 提高命中率)
    search_results = web_search(f"{province} {price_type_name} 元/千瓦时 最新", max_results=5)

    # 第二次: 第一次没结果, 换更宽泛的关键词
    if not search_results:
        logger.info("   第一次无结果，尝试宽泛搜索...")
        search_results = web_search(f"{province} {price_type_name} 电价", max_results=5)

    if search_results:
        # 把搜索结果拼成带编号+来源的文本, 供 LLM 提取
        source_text = "\n\n".join(
            f"[{i}] {r['title']}\n{r['snippet']}\n链接: {r['url']}"
            for i, r in enumerate(search_results, 1)
        )
        logger.info(f"搜索到 {len(search_results)} 条结果")

        # 优先用 LLM 从搜索结果中提取结构化价格 → 入库缓存 → 返回真实价格
        # 这样下次查询直接命中 DB, 实现"越查越快"
        extracted = _extract_price_with_llm(source_text, province, price_type)
        if extracted and extracted.get("price"):
            try:
                price = float(extracted["price"])   # 转 float 入库
                insert_price(
                    province, year_month, price_type, price,   # 唯一键: 省+月+类型
                    unit="元/千瓦时",
                    source=source_text[:500],       # 来源摘要(截断防超字段长度)
                )
                logger.info(f"✅ 提取并入库: {province} {year_month} {price_type_name} = {price} 元/千瓦时")
                return {
                    "province": province, "year_month": year_month, "price_type": price_type,
                    "price_type_name": price_type_name, "price": price,
                    "unit": "元/千瓦时", "source": "实时 Web 搜索 (已缓存)",
                    "cached": False, "trend": [], "search_used": True,
                }
            except Exception as e:
                logger.warning(f"入库失败: {e}")    # 入库失败不阻塞, 走兜底

        # 提取/入库失败 → 退回把搜索结果交给对话层 Agent 提取 (price=-1 特殊标记)
        logger.info("LLM 未能提取价格，退回交给 Agent 提取")
        return {
            "province": province, "year_month": year_month, "price_type": price_type,
            "price_type_name": price_type_name,
            "price": -1,  # 特殊标记：表示"有搜索结果待提取"
            "unit": "元/千瓦时",
            "source": source_text,
            "cached": False, "trend": [], "search_used": True,
        }

    # 4. 搜索也完全没结果 → 空结果
    return _empty_result(province, year_month, price_type, price_type_name, search_used=True)


def _empty_result(province: str, year_month: str, price_type: str,
                  price_type_name: str, search_used: bool = False) -> PriceResult:
    """构造空结果(DB和搜索都无数据)。"""
    return {
        "province": province,
        "year_month": year_month,
        "price_type": price_type,
        "price_type_name": price_type_name,
        "price": None,                            # None 表示完全无数据
        "unit": "元/千瓦时",
        "source": "",
        "cached": False,
        "trend": [],
        "search_used": search_used,
    }
