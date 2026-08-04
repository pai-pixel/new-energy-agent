"""
Web 搜索工具 - DuckDuckGo 搜索 + 页面抓取
用于电价实时查询和新能源政策知识检索
"""

import logging
from typing import TypedDict

import httpx

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    DuckDuckGo 文本搜索
    返回: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
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
        logger.info(f"搜索 '{query[:50]}...' → {len(results)} 条结果")
    except Exception as e:
        logger.warning(f"DuckDuckGo 搜索失败: {e}")
        # 回退: 尝试 requests 直接调用
        try:
            results = _fallback_search(query, max_results)
        except Exception:
            logger.error("回退搜索也失败了")
    return results


def _fallback_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """回退搜索方案"""
    import requests
    url = "https://html.duckduckgo.com/html/"
    resp = requests.get(url, params={"q": query}, timeout=10)
    # 简单 HTML 解析
    results: list[SearchResult] = []
    # 这里用简单的文本分割，避免引入 BeautifulSoup
    from html.parser import HTMLParser

    class DDParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.in_result = False
            self.in_link = False
            self.in_snippet = False
            self.current = {}
            self.data = ""

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                self.in_link = True
                self.current["url"] = attrs_dict.get("href", "")
            if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                self.in_snippet = True

        def handle_data(self, data):
            if self.in_link:
                self.current["title"] = data.strip()
            if self.in_snippet:
                self.current["snippet"] = data.strip()

        def handle_endtag(self, tag):
            if tag == "a" and self.in_link:
                self.in_link = False
            if tag == "a" and self.in_snippet:
                self.in_snippet = False
                if self.current.get("title"):
                    self.results.append(dict(self.current))
                self.current = {}

    parser = DDParser()
    parser.feed(resp.text)
    results = parser.results[:max_results]
    return results


def web_fetch(url: str, timeout: int = 15) -> str:
    """
    抓取网页全文 (markdown 格式)
    优先用 httpx，回退 requests
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewEnergyAgent/1.0; +https://github.com/pai-pixel/new-energy-agent)"
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.warning(f"httpx 抓取 {url} 失败: {e}，尝试 requests")
        import requests
        resp = requests.get(url, headers=headers, timeout=timeout)
        html = resp.text

    # 简单提取文本内容 (去除 script/style 标签)
    text = _extract_text(html)
    # 截断过长的内容 (留给 LLM 处理)
    if len(text) > 8000:
        text = text[:8000] + "\n...(内容过长已截断)"
    return text


def _extract_text(html: str) -> str:
    """从 HTML 中提取有用文本"""
    import re
    # 移除 script 和 style
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', html)
    # 合并空白
    text = re.sub(r'\s+', ' ', text)
    # 解码常见 HTML 实体
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&apos;', "'")
    return text.strip()
