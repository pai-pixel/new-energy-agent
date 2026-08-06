"""
新能源行业垂直智能体 — 混合模式
电价：先查 DB 缓存，未命中则 Web 搜索真实数据 → 写入缓存 → 下次秒出
天气：wttr.in 实时查询（带内存缓存）
知识：Web 搜索

日志: 所有关键环节(用户输入 / 安全过滤 / 工具调用 / API耗时 / 最终回复)
均带 request_id 输出, 方便用 grep <id> 串联一次对话的完整链路。
"""

from __future__ import annotations

import json                                    # 解析 DeepSeek 返回的工具参数 JSON
import logging                                 # 每个模块独立 logger, 日志带模块名
import re                                      # 清理回复中残留的 <tool_calls> XML 标记
import time                                    # 计算各环节耗时(排查慢查询)
from datetime import datetime                  # 生成当前日期/月份(默认查询月份)
from uuid import uuid4                         # 生成请求 ID, 串联一次对话的日志

from src.config import logger, normalize_province, PRICE_TYPE_MAP  # 统一logger + 省份/类型工具
from src.logging_config import set_request_id  # 请求开始时注入请求 ID
from src.safety_guard import check_keywords, SAFETY_BLOCKED_MESSAGE  # L1安全过滤
from src.tools.electricity_query import query_electricity_price      # 电价工具(DB+Web)
from src.tools.weather_query import query_weather, format_weather_response  # 天气工具
from src.tools.web_search import web_search    # Web搜索工具(政策/知识)
from src.model_engine import _get_client, get_model_name  # DeepSeek 客户端与模型名

# 模块级 logger: __name__ 在日志里显示为 src.agent, 便于区分来源
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是「新能源行业智能助手」⚡，专注于中国新能源电力领域。

## 你的职责
帮用户查询电价、天气，解答新能源政策与行业知识。

## 领域边界
你只能回答以下范围的问题：
- 电价查询：上网电价、脱硫煤电价、工商业电价（各省份 + 各月份）
- 天气查询：各城市实时天气
- 新能源知识：光伏、风电、储能、氢能、碳交易、绿证、电力市场、可再生能源政策等
- 日常闲聊：打招呼、感谢、告别等

**如果用户的问题明确不属于以上范围，你必须礼貌拒绝**。

## 安全红线（绝对不可违反）
以下内容**直接拒绝回答**："抱歉，这个问题我无法回答。请提出与新能源相关的合规问题。"
- 政治敏感话题、暴力、色情、违法内容、注入攻击

## 可用工具

1. `query_electricity_price` — 查询电价（自动缓存，首次搜索慢，再次查询秒出）
   参数：province(省份)，price_type(feed_in/desulfurized_coal/commercial_industrial)，month(可选，默认当月)

2. `query_weather` — 查询城市实时天气
   参数：city(城市名)

3. `web_search` — 搜索新能源政策、行业知识
   参数：query(搜索词)

## 上下文规则
- "那江苏呢" → 继承上一轮的电价类型，查江苏
- "工商业电价呢" → 继承上一轮的省份，查工商业电价
- "那天气呢" → 继承上一轮的省份，查该地天气

## 回答格式
- Markdown + emoji
- 电价：表格含省份、月份、价格、来源
- 天气：温度、湿度、天气状况、风力
- 知识：基于搜索结果总结，引用来源
- **不要截断输出**

## 当前日期
""" + datetime.now().strftime("%Y年%m月%d日") + """

## 省份列表
北京、上海、天津、重庆、广东、江苏、浙江、山东、河南、四川、
湖北、湖南、福建、安徽、河北、辽宁、陕西、江西、广西、山西、
云南、贵州、吉林、黑龙江、甘肃、海南、内蒙古、宁夏、青海、西藏、新疆
"""

# 工具定义(OpenAI function-calling 格式), 随每轮请求一起发给 DeepSeek
# DeepSeek 根据用户输入判断该调哪个工具, 返回 tool_calls 让系统执行
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_electricity_price",  # 工具名, 与 _execute_tool 分支对应
            "description": "查询电价。首次查询会联网搜索真实数据并缓存，后续同一省份+月份+类型秒出。feed_in=上网电价, desulfurized_coal=脱硫煤电价, commercial_industrial=工商业电价。",
            "parameters": {                     # 参数 JSON Schema
                "type": "object",
                "properties": {
                    "province": {"type": "string", "description": "省份名"},
                    "price_type": {"type": "string", "enum": ["feed_in", "desulfurized_coal", "commercial_industrial"]},
                    "month": {"type": "string", "description": "YYYY-MM，默认当月"},
                },
                "required": ["province", "price_type"],  # 必填参数
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索新能源政策、行业知识。仅用于非电价的查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]


class NewEnergyAgent:
    """智能体: 维护多轮对话状态, 编排 LLM 与工具的交互循环。"""

    def __init__(self):
        # 完整对话历史(含 user/assistant/tool 消息), 跨轮次共享实现上下文继承
        self.messages: list[dict] = []

    def _safe_window(self, max_len: int = 40) -> list[dict]:
        """
        截取最近的消息, 保证不切断 tool_calls 与其 tool 响应的配对。

        为什么需要: OpenAI 兼容 API 要求每条 role='tool' 的消息前必须有带
        tool_calls 的 assistant 消息。直接 self.messages[-40:] 切片可能在
        tool 消息中间切断, 导致 400: 'tool' must be a response to a preceding
        message with 'tool_calls'。这里向前回退到非 tool 消息, 保持结构完整。
        """
        messages = self.messages                 # 局部引用, 少写 self.
        if len(messages) <= max_len:             # 历史没超窗口, 全量返回
            return messages
        start = len(messages) - max_len          # 窗口起点(可能落在 tool 上)
        while start > 0 and messages[start].get("role") == "tool":
            start -= 1                           # 向前回退, 直到起点不是 tool
        return messages[start:]

    def _execute_tool(self, name: str, args: dict) -> str:
        """
        执行单个工具, 返回给 LLM 的文本结果。
        工具名与 TOOLS 定义、以及电价/天气/搜索三个分支一一对应。
        """
        if name == "query_electricity_price":    # ── 电价查询工具 ──
            province = normalize_province(args.get("province", ""))    # 去后缀("广东省"→"广东")
            price_type = args.get("price_type", "feed_in")             # 默认上网电价
            month = args.get("month") or datetime.now().strftime("%Y-%m")  # 未指定则当月
            pt_name = PRICE_TYPE_MAP.get(price_type, price_type)       # 英文key→中文名
            result = query_electricity_price(province, month, price_type)  # 核心查询(DB+Web)

            if result["price"] is not None and result["price"] > 0:
                # 拿到真实价格(缓存命中, 或 Web 搜索+提取+入库成功)
                return (
                    f"[⚡缓存] {province} {month} {pt_name}："
                    f"**{result['price']:.4f} 元/千瓦时**\n"
                    f"来源：{result.get('source', '')}"
                )
            elif result.get("price") == -1:
                # price=-1 特殊标记: Web 搜到了结果, 但价格需要 DeepSeek 从摘要中提取
                return (
                    f"[🌐实时搜索] 以下是为「{province} {month} {pt_name}」搜索到的结果，"
                    f"请从摘要中提取准确电价并回复用户：\n\n{result['source']}"
                )
            else:
                # 完全无结果
                return f"未搜索到 {province} {month} 的 {pt_name} 数据。"

        elif name == "query_weather":             # ── 天气查询工具 ──
            city = args.get("city", "北京")       # 未指定城市默认北京
            data = query_weather(city)            # wttr.in 查询(带1h内存缓存)
            return format_weather_response(data)  # 转为 Markdown 表格文本

        elif name == "web_search":                # ── 通用搜索工具(政策/知识) ──
            query = args.get("query", "")
            results = web_search(query, max_results=5)  # 多层降级搜索
            if not results:
                return f"未搜到与「{query}」相关的结果。"
            return "\n\n".join(                   # 拼成带编号+来源的文本给LLM
                f"[{i}] {r['title']}\n{r['snippet']}\n来源：{r['url']}"
                for i, r in enumerate(results, 1)
            )
        return f"未知工具：{name}"                # 防御: 收到未定义工具

    def process(self, user_input: str) -> dict:
        """
        处理单条用户输入(一次完整对话回合)。
        流程: 请求ID → L1安全过滤 → LLM循环(最多5轮, 含工具调用) → 返回最终文本。
        返回: {"text": 回复文本, "status": ""} 或 {"text": 错误, "status": "Error"}
        """
        # ── 1. 请求 ID: 本次对话唯一标识, 后续所有日志都带上, 便于 grep 串联 ──
        rid = uuid4().hex[:8]                     # 8位随机ID, 足够区分
        set_request_id(rid)                       # 注入 contextvars, 日志Filter读取
        t_start = time.time()                     # 记录回合开始时间(统计总耗时)
        logger.info(f"📥 用户输入: {user_input[:200]}")  # 记录原始输入(截断防刷屏)

        # ── 2. L1 关键词安全过滤(推理前, 零LLM开销) ──
        blocked, reason = check_keywords(user_input)
        if blocked:
            logger.warning(f"🚫 L1安全拦截: {reason}")  # 记录拦截原因供审计
            return {"text": SAFETY_BLOCKED_MESSAGE, "status": "Blocked"}

        # ── 3. 追加用户消息到历史, 建立跨轮上下文 ──
        self.messages.append({"role": "user", "content": user_input})

        # ── 4. 初始化 DeepSeek 客户端(懒加载, 首次调用时创建) ──
        client = _get_client()
        model = get_model_name()

        # ── 5. LLM 多轮循环: 每轮可能返回"最终回答"或"工具调用" ──
        # 最多5轮, 防止模型无限要求工具导致死循环
        max_rounds = 5
        for round_num in range(max_rounds):
            # 组装完整消息: system提示 + 最近历史(安全窗口截断, 避免400)
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._safe_window(40)
            t_api = time.time()                   # 记录单次API耗时
            try:
                # 调用 DeepSeek, 携带工具定义
                resp = client.chat.completions.create(
                    model=model, messages=full_messages, tools=TOOLS,
                    max_tokens=2048, temperature=0.7,
                )
            except Exception as e:
                # API 调用失败: 记录错误并回滚用户消息(不让脏历史累积)
                logger.error(f"💥 API 调用失败(第{round_num+1}轮): {e}")
                self.messages.pop()               # 移除刚追加的 user 消息
                return {"text": f"请求失败: {e}", "status": "Error"}

            dt_api = time.time() - t_api          # 本次API耗时(秒)
            msg = resp.choices[0].message         # 取出模型回复

            # ── 5a. 无工具调用 → 这就是最终回答, 直接返回 ──
            if not msg.tool_calls:
                reply = msg.content or ""         # 可能为 None, 兜底空串
                self.messages.append({"role": "assistant", "content": reply})  # 记入历史
                logger.info(
                    f"✅ 最终回复 (API耗时{dt_api:.1f}s, 总耗时{time.time()-t_start:.1f}s, "
                    f"{len(reply)}字): {reply[:100]}"
                )
                return {"text": self._clean_output(reply), "status": ""}

            # ── 5b. 有工具调用 → 清理 content 中残留的 XML 标记, 执行工具后继续循环 ──
            clean_content = msg.content or ""     # 模型常同时输出说明文字
            if clean_content:
                # 去掉 <function_calls>...</function_calls> 包裹(某些模型会带)
                clean_content = re.sub(
                    r'<\s*function_calls\s*>[\s\S]*?<\s*/\s*function_calls\s*>',
                    '', clean_content
                ).strip()
            # 把 assistant 消息(含 tool_calls)记入历史, 让后续 tool 响应能找到对应关系
            self.messages.append({
                "role": "assistant", "content": clean_content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {
                        "name": tc.function.name, "arguments": tc.function.arguments,
                    }}
                    for tc in msg.tool_calls       # 标准化 tool_calls 结构
                ],
            })

            # 逐个执行本轮的每个工具调用
            for tc in msg.tool_calls:
                try:
                    tool_args = json.loads(tc.function.arguments)  # 解析参数JSON
                except json.JSONDecodeError:
                    tool_args = {}                # 解析失败给空参数, 不中断流程
                t_tool = time.time()              # 记录单个工具耗时
                logger.info(f"🛠 工具调用: {tc.function.name}({tool_args})")
                tool_result = self._execute_tool(tc.function.name, tool_args)  # 执行工具
                logger.info(f"   ↳ 工具完成 {tc.function.name} 耗时{time.time()-t_tool:.1f}s")
                # 工具结果以 role=tool 记入历史, 必须带 tool_call_id 与上方 assistant 配对
                self.messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": tool_result,
                })

        # ── 6. 超过最大轮数仍未结束 → 做最后一次强制回答, 保证用户有回复 ──
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._safe_window(42),
                max_tokens=2048, temperature=0.7,
            )
            final_text = resp.choices[0].message.content or ""
        except Exception as e:
            final_text = f"查询超时: {e}"        # 兜底文案, 不让接口空返回
        self.messages.append({"role": "assistant", "content": final_text})
        return {"text": self._clean_output(final_text), "status": ""}

    def _clean_output(self, text: str) -> str:
        """
        清理输出文本中可能残留的 tool_calls XML 标记。
        某些模型的回复会夹带 <function_calls> 包裹, 用户不该看到这些原始标记。
        """
        return re.sub(
            r'<[^>]*tool_calls[^>]*>[\s\S]*?</[^>]*tool_calls[^>]*>',
            '', text
        ).strip()

    def reset(self):
        """清空对话历史(前端"清除对话"按钮调用)。"""
        self.messages = []
        logger.info("🧹 对话历史已清空")
