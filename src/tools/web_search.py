"""
Web 搜索 — 多层降级
优先 duckduckgo_search 库（调 DDG 内部 API，稳定），
兜底 httpx 直连 DDG HTML。
"""

import logging
import re
from urllib.parse import unquote
from typing import TypedDict

import httpx

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """多层降级：ddgs（新版，快且准）→ HTML 直连 → duckduckgo_search（旧版）"""
    # 1. ddgs 新版（优先 — 结果质量好）
    r = _search_via_ddgs(query, max_results)
    if r:
        return r
    # 2. HTML 直连 (无依赖)
    r = _search_via_html(query, max_results)
    if r:
        return r
    # 3. duckduckgo_search 旧库
    return _search_via_lib(query, max_results)


def _search_via_ddgs(query: str, max_results: int) -> list[SearchResult]:
    """ddgs 新版库"""
    results: list[SearchResult] = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = (r.get("title") or "").strip()
                href = (r.get("href") or "").strip()
                body = (r.get("body") or "").strip()
                # 过滤：必须至少有标题或链接
                if not title and not href:
                    continue
                results.append({"title": title, "url": href, "snippet": body})
        if results:
            logger.info(f"搜索 '{query[:50]}...' → {len(results)} 条 (ddgs)")
            return results
    except Exception as e:
        logger.warning(f"ddgs 失败: {e}")
    return results


def _search_via_lib(query: str, max_results: int) -> list[SearchResult]:
    """duckduckgo_search / ddgs 库"""
    results: list[SearchResult] = []
    for mod_path in ["duckduckgo_search", "ddgs"]:
        try:
            mod = __import__(mod_path, fromlist=["DDGS"])
            with mod.DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            if results:
                logger.info(f"搜索 '{query[:50]}...' → {len(results)} 条 (via {mod_path})")
                return results
        except Exception:
            continue
    return results


def _decode_ddg_url(raw: str) -> str:
    """解码 DuckDuckGo 重定向 URL"""
    m = re.search(r'uddg=([^&]+)', raw)
    if m:
        return unquote(m.group(1))
    if raw.startswith("http"):
        return raw
    if raw.startswith("//"):
        return "https:" + raw
    return raw


def _search_via_html(query: str, max_results: int) -> list[SearchResult]:
    """httpx 直连 DuckDuckGo HTML 端点，简单稳健的解析"""
    results: list[SearchResult] = []
    try:
        with httpx.Client(timeout=15, follow_redirects=True, headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36"
            ),
        }) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
            resp.raise_for_status()
            html = resp.text

        # 方法：找到所有包含 uddg= 的 <a> 标签，提取 URL 和标题
        # DDG HTML 结果格式: <a rel="nofollow" href="//duckduckgo.com/l/?uddg=URL">标题</a>
        # 后面的 <a class="result__snippet">摘要</a>

        links = re.findall(
            r'<a[^>]*uddg=([^"&]+)[^>]*>([^<]+)</a>',
            html
        )

        snippets = re.findall(
            r'class="result__snippet">(.*?)</a>',
            html, re.DOTALL
        )
        # 清理 snippet
        clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets]

        seen = set()
        snippet_idx = 0
        for raw_url, title in links:
            title = title.strip()
            if not title or len(title) < 3:
                continue
            url = _decode_ddg_url(unquote(raw_url))
            if not url.startswith("http"):
                continue
            if url in seen:
                continue
            seen.add(url)

            snippet = ""
            if snippet_idx < len(clean_snippets):
                snippet = clean_snippets[snippet_idx]
                snippet_idx += 1

            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break

        if results:
            logger.info(f"HTML 搜索 '{query[:50]}...' → {len(results)} 条结果")
        else:
            logger.warning(f"HTML 搜索 '{query[:50]}...' 未提取到结果")

    except Exception as e:
        logger.warning(f"HTML 搜索异常: {e}")

    return results


def web_fetch(url: str, timeout: int = 15) -> str:
    """抓取网页并提取纯文本"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        )
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.warning(f"抓取 {url[:60]} 失败: {e}")
        return ""

    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)
    for ent, ch in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                     ('&quot;', '"'), ('&apos;', "'")]:
        text = text.replace(ent, ch)
    text = text.strip()
    if len(text) > 8000:
        text = text[:8000] + "\n...(截断)"
    return text
