"""
Web 搜索 — 多层降级
优先 ddgs 新版库（结果质量好），
兜底 duckduckgo_search 旧库 → httpx 直连 DDG HTML。

失败链路: ddgs → duckduckgo_search → 抓 HTML
任一层拿到结果即返回, 全部失败返回空列表(调用方自行兜底)。
"""

import logging                                 # 模块 logger
import re                                      # HTML 解析 + URL 解码
from urllib.parse import unquote               # 反转义 DDG 重定向 URL
from typing import TypedDict                   # 类型标注

import httpx                                   # HTML 兜底方案的 HTTP 客户端

# 模块级 logger
logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    """搜索结果结构(类型标注)"""
    title: str       # 标题
    url: str         # 真实链接
    snippet: str     # 摘要


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    多层降级搜索入口。
    query: 搜索词; max_results: 最多返回条数。
    依次尝试: ddgs(新版库) → HTML直连 → duckduckgo_search(旧库)。
    任一成功即返回; 全部失败返回空列表(调用方自行兜底)。
    """
    # 1. ddgs 新版(优先 — 结果质量好, 走官方后端)
    r = _search_via_ddgs(query, max_results)
    if r:
        return r
    # 2. HTML 直连(无第三方依赖, 最稳)
    r = _search_via_html(query, max_results)
    if r:
        return r
    # 3. duckduckgo_search 旧库(最后兜底)
    return _search_via_lib(query, max_results)


def _search_via_ddgs(query: str, max_results: int) -> list[SearchResult]:
    """ddgs 新版库: 结果质量最好的首选方案。"""
    results: list[SearchResult] = []
    try:
        from ddgs import DDGS                   # 新版库, 惰性导入(可能未安装)
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = (r.get("title") or "").strip()    # 标题, 去空白
                href = (r.get("href") or "").strip()      # 链接
                body = (r.get("body") or "").strip()      # 摘要
                # 过滤: 必须至少有标题或链接(排除空结果)
                if not title and not href:
                    continue
                results.append({"title": title, "url": href, "snippet": body})
        if results:
            logger.info(f"搜索 '{query[:50]}...' → {len(results)} 条 (ddgs)")
            return results
    except Exception as e:
        logger.warning(f"ddgs 失败: {e}")       # 降级到下一层
    return results


def _search_via_lib(query: str, max_results: int) -> list[SearchResult]:
    """
    duckduckgo_search / ddgs 旧库。
    依次尝试导入这两个库, 名字可能随版本变化, 都试试。
    """
    results: list[SearchResult] = []
    for mod_path in ["duckduckgo_search", "ddgs"]:  # 两个可能的模块名
        try:
            mod = __import__(mod_path, fromlist=["DDGS"])  # 动态导入
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
            continue                            # 该库不可用, 试下一个
    return results


def _decode_ddg_url(raw: str) -> str:
    """
    解码 DuckDuckGo 重定向 URL。
    DDG 结果链接形如 //duckduckgo.com/l/?uddg=<编码后的真实URL>,
    需要取出 uddg 参数并反转义得到真实地址。
    """
    m = re.search(r'uddg=([^&]+)', raw)          # 提取 uddg 参数
    if m:
        return unquote(m.group(1))               # 反转义百分号编码
    if raw.startswith("http"):                   # 已是绝对 URL
        return raw
    if raw.startswith("//"):                     # 协议相对 URL → 补 https:
        return "https:" + raw
    return raw                                   # 兜底原样返回


def _search_via_html(query: str, max_results: int) -> list[SearchResult]:
    """
    httpx 直连 DuckDuckGo HTML 端点, 用正则解析结果。
    依赖最少(只需 httpx), 作为库都失效时的最后保底。
    """
    results: list[SearchResult] = []
    try:
        # 模拟浏览器 UA, 避免被 DDG 拒绝; 15s 超时 + 跟随重定向
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

        # 解析 HTML 结果页:
        # 每个结果形如 <a rel="nofollow" href="//duckduckgo.com/l/?uddg=URL">标题</a>
        # 摘要另在一个 class="result__snippet" 的 <a> 里
        # 用两个正则分别抓"链接+标题"和"摘要", 再按下标对齐
        links = re.findall(
            r'<a[^>]*uddg=([^"&]+)[^>]*>([^<]+)</a>',
            html
        )

        snippets = re.findall(
            r'class="result__snippet">(.*?)</a>',
            html, re.DOTALL
        )
        # 清理 snippet 中的残留 HTML 标签
        clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets]

        seen = set()                             # 去重: 同一 URL 只保留首个
        snippet_idx = 0                          # 摘要与链接按下标配对
        for raw_url, title in links:
            title = title.strip()
            if not title or len(title) < 3:      # 过滤过短/空标题
                continue
            url = _decode_ddg_url(unquote(raw_url))  # 还原真实 URL
            if not url.startswith("http"):       # 过滤非 http 链接
                continue
            if url in seen:                      # 跳过重复
                continue
            seen.add(url)

            snippet = ""
            if snippet_idx < len(clean_snippets):  # 还有摘要可配对
                snippet = clean_snippets[snippet_idx]
                snippet_idx += 1

            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:      # 达到目标条数就停
                break

        if results:
            logger.info(f"HTML 搜索 '{query[:50]}...' → {len(results)} 条结果")
        else:
            logger.warning(f"HTML 搜索 '{query[:50]}...' 未提取到结果")

    except Exception as e:
        logger.warning(f"HTML 搜索异常: {e}")    # 网络失败静默, 返回空列表

    return results


def web_fetch(url: str, timeout: int = 15) -> str:
    """
    抓取网页并提取纯文本(供网页内容分析用, 当前主链路未直接调用)。
    去掉 script/style/标签, 压缩空白, HTML 实体转义。
    """
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

    # 依次清理: script/style → 剩余标签 → 连续空白
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)         # 标签替换为空格(防止粘连)
    text = re.sub(r'\s+', ' ', text)             # 多空白压缩为单空格
    # HTML 实体反转义
    for ent, ch in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                     ('&quot;', '"'), ('&apos;', "'")]:
        text = text.replace(ent, ch)
    text = text.strip()
    if len(text) > 8000:                         # 超长截断, 防止上下文爆炸
        text = text[:8000] + "\n...(截断)"
    return text
