"""
统一日志配置模块 — 所有模块的日志从这里获取格式和输出。

为什么需要它:
- 之前每个文件各自 getLogger, 输出只有控制台, 后台运行(如 nohup)时日志散落难查
- 现在统一: 控制台 + 文件双输出, 文件按大小滚动, 排查时翻 logs/agent.log 即可
- 每次对话生成一个请求 ID (request_id), 自动注入到该请求产生的每条日志里,
  可以用 grep 一个 ID 把"安全过滤 → 工具调用 → 入库 → 最终回复"整条链路串起来

用法:
    from src.logging_config import logger, set_request_id
    logger.info("...")          # 自动带时间/级别/请求ID/模块名
    set_request_id("a1b2c3")    # 请求开始时调用, 之后日志都会带上这个 ID
"""

from __future__ import annotations

import logging
import logging.handlers
from contextvars import ContextVar
from pathlib import Path

# ── 请求 ID 上下文变量 ──────────────────────────────────────────
# contextvars 提供"每个请求独立的值": 即使多个请求并发处理,
# 每个请求内的代码读到的 request_id_var 都是它自己设置的那个值。
# default="-" 表示没有显式设置时的兜底值(如启动阶段的日志)。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: str) -> None:
    """
    在请求开始时设置请求 ID。
    之后该请求生命周期内的所有日志都会自动携带这个 ID, 便于按请求串联排查。
    并发安全: 每个请求在独立的上下文里设置, 互不干扰。
    """
    request_id_var.set(rid)


def get_request_id() -> str:
    """读取当前请求 ID(调试/上报用)。"""
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """
    logging.Filter 的钩子: 每条日志在输出前都会调用 filter(record)。
    我们利用它把 request_id_var 的当前值注入到 record.request_id 属性,
    然后日志格式 %(request_id)s 就能把它渲染出来。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True  # 返回 True = 该日志放行, 不拦截


# ── 日志目录与文件 ──────────────────────────────────────────────
# 文件放在项目根目录下 logs/ 子目录, 与代码目录分离。
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent.log"

# 统一格式: 时间 | 级别 | 请求ID | 来源模块 | 消息
# levelname 固定宽度右对齐(-7s), 级别列对齐更美观
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | req=%(request_id)s | %(name)s | %(message)s"
_formatter = logging.Formatter(_LOG_FORMAT)


def _build_handlers() -> list[logging.Handler]:
    """
    构建双输出 handler:
    1. 控制台 StreamHandler — 终端实时可见(面试演示、本地调试)
    2. 文件 RotatingFileHandler — 按 10MB 滚动, 保留 5 个历史文件, 日志不会无限膨胀
    """
    # 控制台 handler
    console = logging.StreamHandler()
    console.setFormatter(_formatter)
    console.addFilter(RequestIdFilter())

    # 文件 handler: maxBytes=10MB 触发滚动, backupCount=5 保留最近 5 个文件
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(_formatter)
    file_handler.addFilter(RequestIdFilter())
    return [console, file_handler]


def init_logging() -> None:
    """
    初始化根 logger: 设置级别 + 挂载双输出 handler。
    幂等: 重复调用不会重复添加 handler(启动多次或 import 多次都安全)。
    """
    root = logging.getLogger()  # 根 logger
    root.setLevel(logging.INFO)  # 生产/演示默认 INFO, 想看更细可改 DEBUG

    # 第三方 HTTP 库日志降噪: httpx / primp 等每发一个请求就打一条 INFO,
    # 且不带 request_id, 会淹没业务日志。统一压到 WARNING(只保留真实异常)。
    for noisy in ("httpx", "httpcore", "primp", "urllib3", "ddgs", "duckduckgo_search"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 避免重复挂 handler: 用已挂载的 RotatingFileHandler 作为"是否已初始化"的标志
    already = any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
    )
    if already:
        return
    for h in _build_handlers():
        root.addHandler(h)


# ── 模块级 logger ──────────────────────────────────────────────
# 所有模块统一从这里拿 logger: `from src.logging_config import logger`
# __name__ 是各模块自己的名字(如 src.agent), 日志里能区分来源。
logger = logging.getLogger("new_energy_agent")

# 模块加载即初始化, 保证任何地方 import 本模块后日志立刻可用
init_logging()
