"""
新能源行业垂直智能体 - 主入口
Agent 主循环 + Gradio Chat UI + 流式推理
"""

import logging
import time
from datetime import datetime

import gradio as gr
import plotly.graph_objects as go
from openai import OpenAI

from src.config import (
    VLLM_BASE_URL, VLLM_PORT, MODEL_MAX_LEN,
    PRICE_TYPE_MAP, get_city_for_province, normalize_province,
    logger,
)
from src.context_manager import ContextManager, ConversationState
from src.intent_router import IntentRouter
from src.safety_guard import (
    check_keywords, check_domain_boundary,
    is_safety_blocked, is_domain_blocked,
    SAFETY_BLOCKED_MESSAGE, DOMAIN_BOUNDARY_MESSAGE,
)
from src.prompt_templates import FINAL_RESPONSE_SYSTEM, KNOWLEDGE_QUERY_SYSTEM
from src.tools.electricity_query import query_electricity_price
from src.tools.weather_query import query_weather, format_weather_response
from src.tools.web_search import web_search
from src.visualization.chart_builder import build_trend_chart, build_comparison_chart
from src.visualization.table_builder import build_price_table

# ── Agent 核心 ─────────────────────────────────────────────────


class NewEnergyAgent:
    """新能源行业智能体 - 编排层"""

    def __init__(self):
        self.context_mgr = ContextManager()
        self.intent_router = IntentRouter()
        self.state = ConversationState()
        self.llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
        self.last_chart: go.Figure | None = None
        self.last_table: "pd.DataFrame | None" = None

    def process(self, user_input: str) -> dict:
        """
        处理用户消息的主流程

        Returns:
            {"text": str, "chart": go.Figure | None, "table": pd.DataFrame | None, "status": str}
        """
        self.last_chart = None
        self.last_table = None

        # ─── Step 1: 第一层安全过滤 (关键词) ───
        blocked, reason = check_keywords(user_input)
        if blocked:
            logger.info(f"🛡️ 第一层拦截: {reason}")
            return self._result(SAFETY_BLOCKED_MESSAGE, status="🛡️ 已拦截")

        # ─── Step 2: 意图分类 + 实体抽取 (LLM) ───
        try:
            intent_result = self.intent_router.classify(user_input, self.state)
        except Exception as e:
            logger.error(f"意图分类失败: {e}")
            return self._result(
                "抱歉，我暂时无法理解你的问题，请稍后再试或换一种说法。",
                status="⚠️ 分类异常"
            )

        intent = intent_result["intent"]
        entities = intent_result["entities"]
        inherit = intent_result["inherit_from_context"]
        reasoning = intent_result.get("reasoning", "")

        logger.info(f"🎯 意图: {intent} | 实体: {entities} | 继承: {inherit} | {reasoning}")

        # ─── Step 3: 第三层安全过滤 (LLM 安全标记 + 领域边界) ───
        # LLM 自主判断的安全拒答
        if intent == "out_of_domain":
            _, msg = check_domain_boundary("out_of_domain")
            return self._result(msg, status="📍 领域外")

        # ─── Step 4: 上下文状态更新 ───
        self.state = self.context_mgr.update(self.state, intent, entities, inherit)

        # ─── Step 5: 工具执行 ───
        if intent == "price_query":
            result = self._handle_price_query(entities)
        elif intent == "weather_query":
            result = self._handle_weather_query(entities)
        elif intent == "knowledge_query":
            result = self._handle_knowledge_query(user_input)
        else:  # chat
            result = self._handle_chat(user_input)

        status = self.context_mgr.get_status_text(self.state)
        result["status"] = status
        return result

    def _result(self, text: str, chart=None, table=None, status: str = "") -> dict:
        self.last_chart = chart
        self.last_table = table
        return {"text": text, "chart": chart, "table": table, "status": status}

    # ─── 电价查询处理 ───

    def _handle_price_query(self, entities: dict) -> dict:
        province = entities.get("province") or self.state.current_province
        price_type = entities.get("price_type") or self.state.current_price_type
        year_month = entities.get("month") or datetime.now().strftime("%Y-%m")

        if not province:
            return self._result(
                "请告诉我你想查询哪个省份的电价？比如「上海上网电价」或「江苏脱硫煤电价」。",
                status="⚠️ 缺少省份"
            )
        if not price_type:
            return self._result(
                f"你想查询 **{province}** 的哪种电价呢？\n"
                f"📊 上网电价 | ⚡ 脱硫煤电价 | 🏭 工商业电价",
                status="⚠️ 缺少电价类型"
            )

        province = normalize_province(province)
        price_type_name = PRICE_TYPE_MAP.get(price_type, price_type)

        # 执行查询 (缓存优先 + Web Search 实时)
        start = time.time()
        price_result = query_electricity_price(province, year_month, price_type)
        elapsed = time.time() - start

        # 构建图表
        chart = None
        table = None
        trend = price_result.get("trend", [])

        if trend and len(trend) >= 2:
            chart = build_trend_chart(trend, province, price_type)

        # 构建回复文本
        if price_result["price"] is not None:
            cache_label = "⚡ 缓存" if price_result["cached"] else f"🌐 实时查询 ({elapsed:.1f}s)"
            source_info = f"\n\n📎 来源: {price_result['source']}" if price_result.get("source") else ""
            table = build_price_table(
                [{"year_month": r["year_month"], "price": r["price"]}
                 for r in trend] if trend else [price_result],
                price_type
            )

            text = (
                f"## 📊 {province} {price_type_name}\n\n"
                f"| 项目 | 详情 |\n"
                f"|------|------|\n"
                f"| 🏷️ 省份 | {province} |\n"
                f"| 📅 月份 | {year_month} |\n"
                f"| 💰 电价 | **{price_result['price']:.4f} {price_result['unit']}** |\n"
                f"| 🔖 类型 | {price_type_name} |\n"
                f"| 🏷️ 状态 | {cache_label} |\n"
                f"{source_info}\n"
            )
        else:
            # 搜索了但没提取到价格
            if price_result.get("search_used"):
                text = (
                    f"## ⚠️ 未找到明确电价\n\n"
                    f"在公开渠道中，我暂未找到 **{province}** 的 **{year_month} {price_type_name}**。\n\n"
                    f"建议:\n"
                    f"- 尝试更换查询月份\n"
                    f"- 访问 {province} 发改委或电力交易中心官网\n"
                    f"- 试试搜索「{province} {price_type_name} 最新」"
                )
            else:
                text = f"⚠️ 暂未找到 {province} 的 {price_type_name} 数据，请稍后再试。"

        return self._result(text, chart, table)

    # ─── 天气查询处理 ───

    def _handle_weather_query(self, entities: dict) -> dict:
        city = entities.get("city") or self.state.current_city

        if not city and self.state.current_province:
            city = get_city_for_province(self.state.current_province)

        if not city:
            return self._result(
                "请告诉我你想查询哪个城市的天气？比如「北京天气」",
                status="⚠️ 缺少城市"
            )

        weather_data = query_weather(city)
        text = format_weather_response(weather_data)
        return self._result(text)

    # ─── 知识查询处理 ───

    def _handle_knowledge_query(self, user_input: str) -> dict:
        """新能源知识/政策查询 - 先搜索，再让 LLM 综合回答"""
        # 1. 搜索
        search_query = f"新能源 {user_input}"
        results = web_search(search_query, max_results=5)

        if not results:
            # 无搜索结果，直接用 LLM 知识回答
            try:
                resp = self.llm_client.chat.completions.create(
                    model="qwen",
                    messages=[
                        {"role": "system", "content": FINAL_RESPONSE_SYSTEM},
                        {"role": "user", "content": user_input},
                    ],
                    max_tokens=1024,
                    temperature=0.7,
                )
                return self._result(resp.choices[0].message.content)
            except Exception as e:
                return self._result(f"抱歉，搜索和回答暂时不可用: {e}")

        # 2. 格式化搜索结果
        search_text_parts = []
        for i, r in enumerate(results, 1):
            search_text_parts.append(
                f"[{i}] {r['title']}\n{r['snippet']}\n来源: {r['url']}"
            )
        search_text = "\n\n".join(search_text_parts)

        # 3. LLM 综合回答
        system_prompt = KNOWLEDGE_QUERY_SYSTEM.replace("{search_results}", search_text)

        try:
            resp = self.llm_client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            text = resp.choices[0].message.content
            text += f"\n\n---\n📚 *以上信息基于联网搜索，仅供参考*"
            return self._result(text)
        except Exception as e:
            # 回退：直接返回搜索结果
            return self._result(
                f"## 🔍 关于「{user_input}」的搜索结果\n\n{search_text}\n\n"
                f"---\n⚠️ AI 回答生成失败: {e}",
            )

    # ─── 闲聊处理 ───

    def _handle_chat(self, user_input: str) -> dict:
        try:
            resp = self.llm_client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": FINAL_RESPONSE_SYSTEM},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=512,
                temperature=0.8,
            )
            return self._result(resp.choices[0].message.content)
        except Exception as e:
            return self._result(
                f"你好！我是新能源行业智能助手 ⚡\n\n"
                f"我可以帮你:\n"
                f"📊 查询各省上网电价\n"
                f"⚡ 查询各省脱硫煤电价\n"
                f"🏭 查询各省工商业电价\n"
                f"🌤️ 查询城市天气\n"
                f"📚 解答新能源政策与行业知识\n\n"
                f"请问有什么可以帮你的？"
            )

    def reset(self):
        """重置对话"""
        self.state = self.context_mgr.reset()
        self.last_chart = None
        self.last_table = None


# ── Gradio UI ──────────────────────────────────────────────────

# CSS 样式
CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.85em;
    padding: 10px;
}
.status-bar {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.9em;
    color: #166534;
    margin: 8px 0;
}
.chatbot {
    border-radius: 12px !important;
}
"""


def create_ui(agent: NewEnergyAgent) -> gr.Blocks:
    """构建 Gradio UI"""

    with gr.Blocks(title="新能源行业智能助手") as demo:

        gr.Markdown(
            """# ⚡ 新能源行业智能助手
            📊 上网电价 · ⚡ 脱硫煤电价 · 🏭 工商业电价 · 🌤️ 天气 · 📚 政策咨询
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_classes=["chatbot"],
                    height=500,
                    avatar_images=(None, "⚡"),
                    render_markdown=True,
                )
            with gr.Column(scale=3):
                plot_output = gr.Plot(label="📈 电价趋势", visible=True)
                table_output = gr.Dataframe(
                    label="📋 数据明细",
                    visible=True,
                    wrap=True,
                )

        status_bar = gr.Markdown(
            "📍 等待输入... | 第 0 轮",
            elem_classes=["status-bar"],
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="输入问题... 如: 上海上网电价 / 江苏脱硫煤电价 / 北京天气 / 光伏补贴政策",
                scale=5,
                show_label=False,
                container=False,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
            clear_btn = gr.Button("清空", variant="secondary", scale=1)

        # 快捷按钮
        with gr.Row():
            gr.Markdown("**快捷查询:**")
        with gr.Row():
            btn_feed_in = gr.Button("📊 上网电价", size="sm", scale=1)
            btn_coal = gr.Button("⚡ 脱硫煤电价", size="sm", scale=1)
            btn_commercial = gr.Button("🏭 工商业电价", size="sm", scale=1)
            btn_weather = gr.Button("🌤️ 天气", size="sm", scale=1)
            btn_knowledge = gr.Button("📚 政策咨询", size="sm", scale=1)

        # 示例
        gr.Examples(
            examples=[
                "上海上网电价",
                "江苏脱硫煤电价",
                "浙江工商业电价",
                "北京天气怎么样",
                "光伏补贴最新政策",
                "碳中和是什么意思",
            ],
            inputs=msg_input,
        )

        gr.Markdown(
            '<div class="footer">⚡ 新能源行业智能助手 | 数据来源: 公开电价信息 & wttr.in | AI 生成内容仅供参考</div>'
        )

        # ── 事件处理 ──

        def on_message(message: str, history: list):
            """处理用户消息并流式更新"""
            if not message.strip():
                yield history, gr.update(), gr.update(), "📍 请输入问题"
                return

            # 处理消息
            result = agent.process(message.strip())

            # 更新聊天历史
            history.append([message, result["text"]])

            # 更新图表和表格
            chart_update = result["chart"] if result["chart"] else gr.update(visible=False)
            table_update = result["table"] if result["table"] is not None else gr.update(visible=False)

            if result["chart"]:
                chart_update = gr.update(value=result["chart"], visible=True)
            if result["table"] is not None:
                table_update = gr.update(value=result["table"], visible=True)

            yield history, chart_update, table_update, result.get("status", "")

        def on_quick_action(action: str, history: list):
            """快捷按钮填充输入"""
            return action

        def on_clear():
            agent.reset()
            return [], gr.update(visible=False), gr.update(visible=False), "📍 等待输入... | 第 0 轮"

        # 绑定事件
        msg_input.submit(
            on_message, [msg_input, chatbot],
            [chatbot, plot_output, table_output, status_bar]
        ).then(lambda: "", None, msg_input)

        send_btn.click(
            on_message, [msg_input, chatbot],
            [chatbot, plot_output, table_output, status_bar]
        ).then(lambda: "", None, msg_input)

        clear_btn.click(on_clear, None, [chatbot, plot_output, table_output, status_bar])

        # 快捷按钮事件
        def add_prefix(prefix):
            return prefix + " "

        btn_feed_in.click(lambda: "上网电价 ", None, msg_input)
        btn_coal.click(lambda: "脱硫煤电价 ", None, msg_input)
        btn_commercial.click(lambda: "工商业电价 ", None, msg_input)
        btn_weather.click(lambda: "天气 ", None, msg_input)
        btn_knowledge.click(lambda: "", None, msg_input)  # 让用户自行输入

    return demo


# ── 主入口 ─────────────────────────────────────────────────────

def wait_for_vllm(max_retries: int = 60, interval: int = 5):
    """等待 vLLM 服务就绪"""
    from openai import OpenAI
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    for i in range(max_retries):
        try:
            models = client.models.list()
            logger.info(f"✅ vLLM 就绪，可用模型: {[m.id for m in models]}")
            return True
        except Exception:
            logger.info(f"⏳ 等待 vLLM... ({i+1}/{max_retries})")
            time.sleep(interval)
    logger.error("❌ vLLM 启动超时")
    return False


def start_ngrok(port: int = 7860):
    """启动 ngrok 隧道"""
    from src.config import NGROK_TOKEN
    if not NGROK_TOKEN:
        logger.warning("未配置 NGROK_TOKEN，跳过 ngrok")
        return None
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = ngrok.connect(port)
        logger.info(f"🔗 ngrok 隧道: {tunnel.public_url}")
        return tunnel.public_url
    except Exception as e:
        logger.warning(f"ngrok 启动失败: {e}")
        return None


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚡ 新能源行业智能体 启动中...")
    logger.info("=" * 60)

    # 等待 vLLM
    if not wait_for_vllm():
        logger.error("vLLM 未就绪，请先启动 vLLM 服务")
        exit(1)

    # 创建 Agent
    agent = NewEnergyAgent()
    logger.info("✅ Agent 初始化完成")

    # 启动 ngrok
    ngrok_url = start_ngrok(7860)

    # 启动 Gradio
    demo = create_ui(agent)
    logger.info("🚀 启动 Gradio UI...")
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="green"),
    )
