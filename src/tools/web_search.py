"""
Web 搜索工具 - httpx 直连 DuckDuckGo HTML
用于电价实时查询和新能源政策知识检索
"""

import logging
import re
from typing import TypedDict
from urllib.parse import unquote

import httpx

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    DuckDuckGo 搜索
    1. 优先: httpx 直连 html.duckduckgo.com
    2. 兜底: duckduckgo_search 库
    """
    results = _search_via_html(query, max_results)
    if results:
        return results

    results = _search_via_lib(query, max_results)
    if results:
        return results

    logger.warning(f"所有搜索方式均无结果: {query}")
    return []


def _search_via_html(query: str, max_results: int) -> list[SearchResult]:
    """httpx 直连 DuckDuckGo HTML 端点"""
    results: list[SearchResult] = []
    try:
        with httpx.Client(timeout=15, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
            resp.raise_for_status()
            html = resp.text

        # 提取结果: class="result__body" 包装每个结果
        result_blocks = re.findall(
            r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html, re.DOTALL
        )
        # 更宽松的匹配
        if not result_blocks:
            result_blocks = re.findall(
                r'class="result__body">(.*?)<div class="result__extras',
                html, re.DOTALL
            )

        for block in result_blocks:
            # 提取标题和链接
            title_match = re.search(
                r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                block, re.DOTALL
            )
            if not title_match:
                continue

            raw_url = title_match.group(1)
            title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()

            # 解码 DuckDuckGo 重定向 URL
            url = _decode_ddg_url(raw_url)

            # 提取摘要
            snippet = ""
            snippet_match = re.search(
                r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                block, re.DOTALL
            )
            if snippet_match:
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()

            if title and url and url not in {r["url"] for r in results}:
                results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= max_results:
                    break

        if results:
            logger.info(f"搜索 '{query[:50]}...' → {len(results)} 条结果")
    except Exception as e:
        logger.debug(f"HTML 搜索失败: {e}")

    return results


def _search_via_lib(query: str, max_results: int) -> list[SearchResult]:
    """duckduckgo_search 库兜底"""
    results: list[SearchResult] = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        if results:
            logger.info(f"lib 搜索 '{query[:50]}...' → {len(results)} 条结果")
    except Exception:
        # 尝试 ddgs 新库
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            if results:
                logger.info(f"ddgs 搜索 '{query[:50]}...' → {len(results)} 条结果")
        except Exception:
            pass
    return results


def _decode_ddg_url(raw_url: str) -> str:
    """解码 DuckDuckGo 重定向 URL"""
    # //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
    if "uddg=" in raw_url:
        match = re.search(r'uddg=([^&]+)', raw_url)
        if match:
            return unquote(match.group(1))
    # 直接 http(s) 链接
    if raw_url.startswith("http"):
        return raw_url
    # // 开头的链接
    if raw_url.startswith("//"):
        return "https:" + raw_url
    return raw_url


def web_fetch(url: str, timeout: int = 15) -> str:
    """
    抓取网页全文文本
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.warning(f"抓取 {url} 失败: {e}")
        return ""

    text = _extract_text(html)
    if len(text) > 8000:
        text = text[:8000] + "\n...(内容过长已截断)"
    return text


def _extract_text(html: str) -> str:
    """从 HTML 中提取有用文本"""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&apos;', "'")
    return text.strip()
