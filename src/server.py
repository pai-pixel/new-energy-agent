"""
新能源智能体 - FastAPI 后端
提供 HTTP 接口给前端聊天页调用:
- POST /api/chat   对话(单轮), 内部复用 NewEnergyAgent 的多轮循环
- POST /api/reset  清空对话历史
- GET  /           返回静态聊天页 ui/index.html
- GET  /api/logs   返回日志(按请求ID过滤或最近N条), 供前端"查看日志"按钮调用
"""

from __future__ import annotations

import logging                                  # 模块 logger
from pathlib import Path                        # 定位 UI 静态文件

from fastapi import FastAPI, HTTPException       # Web 框架 + HTTP 错误
from fastapi.responses import HTMLResponse       # 返回 HTML 页面
from pydantic import BaseModel                   # 请求/响应体校验

from src.agent import NewEnergyAgent             # 智能体核心
from src.logging_config import init_logging, get_request_id, LOG_FILE  # 日志: 初始化/读请求ID/日志文件路径

# 模块级 logger(__name__ = src.server)
logger = logging.getLogger(__name__)

# 启动即初始化日志系统(控制台+文件+请求ID), 幂等
init_logging()

# FastAPI 应用实例; docs_url=None 关闭 /docs, 面试演示不暴露接口文档
app = FastAPI(title="New Energy Agent", docs_url=None, redoc_url=None)

# 全局唯一智能体实例: 消息历史存在内存中, 跨请求保持(多用户共享, 演示场景足够)
agent = NewEnergyAgent()
# UI 静态文件目录: 项目根目录下的 ui/
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class ChatRequest(BaseModel):
    """请求体: 只需一个 message 字段(FastAPI 自动校验 JSON)"""
    message: str


class ChatResponse(BaseModel):
    """响应体: text 为回复, status 为空字符串=正常, 'Error'/'Blocked' 表示异常,
    request_id 为本次对话的请求ID(前端据此查询该次日志)"""
    text: str
    status: str = ""
    request_id: str = ""


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    对话端点。
    每个请求生成独立 request_id(在 agent.process 内部设置), 日志自动串联。
    空消息返回 400; 内部异常返回 500。
    """
    if not req.message.strip():                 # 空白消息直接拒绝
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        # 调用智能体处理; process 内部会设置请求 ID 并输出带 ID 的日志
        result = agent.process(req.message.strip())
        # get_request_id() 读取 process 内部设置的本次请求ID, 返回给前端
        return ChatResponse(
            text=result["text"],
            status=result.get("status", ""),
            request_id=get_request_id(),
        )
    except Exception as e:
        # 兜底: 未捕获异常记录日志并返回 500(前端能提示, 后台能排查)
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def get_logs(req: str = "", limit: int = 100):
    """
    日志查询端点(前端"查看日志"按钮调用)。
    - req 指定请求ID → 只返回该次对话的日志
    - req 为空 → 返回最近 limit 条日志
    limit 限制在 1~500, 防止一次拉太多。
    """
    limit = max(1, min(limit, 500))               # 钳制范围, 防滥用
    try:
        with open(LOG_FILE, encoding="utf-8") as f:  # 读取日志文件
            lines = f.readlines()
    except FileNotFoundError:
        return {"logs": [], "request_id": req}    # 日志文件还没生成
    if req:                                        # 按请求ID过滤
        filtered = [l.rstrip("\n") for l in lines if f"req={req}" in l]
        return {"logs": filtered[-limit:], "request_id": req}
    # 全部日志: 取最近 limit 条
    return {"logs": [l.rstrip("\n") for l in lines][-limit:], "request_id": ""}


@app.post("/api/reset")
async def reset():
    """清空对话历史(前端"清除对话"按钮调用)。"""
    agent.reset()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回聊天页面; 文件不存在时给个占位提示。"""
    html_path = UI_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>")


if __name__ == "__main__":
    import uvicorn
    # 启动横幅, 醒目提示服务已就绪
    logger.info("=" * 50)
    logger.info("New Energy Agent starting (DeepSeek-native)...")
    logger.info("=" * 50)
    # host=0.0.0.0 允许局域网访问; port=7860 为 Gradio 惯例端口, 与 README 一致
    uvicorn.run("src.server:app", host="0.0.0.0", port=7860, reload=False, log_level="info")
