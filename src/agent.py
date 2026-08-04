"""
新能源行业垂直智能体 - 主入口
Agent 主循环 + Gradio Chat UI
"""

import logging
import time
from datetime import datetime

import gradio as gr
import plotly.graph_objects as go

from src.config import (
    PRICE_TYPE_MAP, get_city_for_province, normalize_province, NGROK_TOKEN,
    logger,
)
from src.context_manager import ContextManager, ConversationState
from src.intent_router import IntentRouter
from src.safety_guard import (
    check_keywords, check_domain_boundary,
    SAFETY_BLOCKED_MESSAGE,
)
from src.prompt_templates import FINAL_RESPONSE_SYSTEM, KNOWLEDGE_QUERY_SYSTEM
from src.tools.electricity_query import query_electricity_price
from src.tools.weather_query import query_weather, format_weather_response
from src.tools.web_search import web_search
from src.visualization.chart_builder import build_trend_chart, build_comparison_chart
from src.visualization.table_builder import build_price_table
from src.model_engine import generate, generate_json, generate_stream, load_model, is_loaded

# ── Agent 核心 ─────────────────────────────────────────────────


class NewEnergyAgent:
    """新能源行业智能体 - 编排层"""

    def __init__(self):
        # Load model on first LLM call (lazy), not at init
        self.context_mgr = ContextManager()
        self.intent_router = IntentRouter()
        self.state = ConversationState()
        self.last_chart: go.Figure | None = None
        self.last_table: "pd.DataFrame | None" = None

    def process_stream(self, user_input: str):
        self.last_chart = None
        self.last_table = None

        # ─── L1: Keyword safety filter ───
        blocked, reason = check_keywords(user_input)
        if blocked:
            logger.info(f"L1 block: {reason}")
            yield self._result(SAFETY_BLOCKED_MESSAGE, status="Blocked")
            return

        # ─── L2: Intent classification + entity extraction ───
        try:
            intent_result = self.intent_router.classify(user_input, self.state)
        except Exception as e:
            logger.error(f"Intent classify failed: {e}")
            yield self._result("Sorry, please try again.", status="Error")
            return

        intent = intent_result["intent"]
        entities = intent_result["entities"]
        inherit = intent_result["inherit_from_context"]
        logger.info(f"Intent: {intent} entities: {entities} inherit: {inherit}")

        # ─── L3: Domain boundary ───
        if intent == "out_of_domain":
            _, msg = check_domain_boundary("out_of_domain")
            yield self._result(msg, status="Out of domain")
            return

        # ─── Update context state ───
        self.state = self.context_mgr.update(self.state, intent, entities, inherit)
        status_text = self.context_mgr.get_status_text(self.state)

        # ─── Route to handler ───
        if intent == "price_query":
            res = self._handle_price(entities)
            res["status"] = status_text
            yield res
        elif intent == "weather_query":
            res = self._handle_weather(entities)
            res["status"] = status_text
            yield res
        elif intent == "knowledge_query":
            yield from self._handle_knowledge_stream(user_input, status_text)
        else:
            yield from self._handle_chat_stream(user_input, status_text)

    def _result(self, text: str, chart=None, table=None, status: str = "") -> dict:
        self.last_chart = chart
        self.last_table = table
        return {"text": text, "chart": chart, "table": table, "status": status}

    # ─── Price query ───

    def _handle_price(self, entities: dict) -> dict:
        province = entities.get("province") or self.state.current_province
        price_type = entities.get("price_type") or self.state.current_price_type
        year_month = entities.get("month") or datetime.now().strftime("%Y-%m")

        if not province:
            return self._result("Which province? e.g. 'Shanghai feed-in tariff'", status="Missing province")
        if not price_type:
            return self._result(
                f"What type for **{province}**?\nFeed-in / Desulfurized coal / Commercial & industrial",
                status="Missing price type"
            )

        province = normalize_province(province)
        pt_name = PRICE_TYPE_MAP.get(price_type, price_type)

        start = time.time()
        price_result = query_electricity_price(province, year_month, price_type)
        elapsed = time.time() - start

        chart = None
        table = None
        trend = price_result.get("trend", [])

        if trend and len(trend) >= 2:
            chart = build_trend_chart(trend, province, price_type)

        if price_result["price"] is not None:
            cache_label = "Cached" if price_result["cached"] else f"Live ({elapsed:.1f}s)"
            source_info = f"\n\nSource: {price_result['source']}" if price_result.get("source") else ""
            if trend:
                table = build_price_table(
                    [{"year_month": r["year_month"], "price": r["price"]} for r in trend],
                    price_type
                )
            text = (
                f"## {province} {pt_name}\n\n"
                f"| Item | Detail |\n|------|--------|\n"
                f"| Province | {province} |\n"
                f"| Month | {year_month} |\n"
                f"| Price | **{price_result['price']:.4f} {price_result['unit']}** |\n"
                f"| Type | {pt_name} |\n"
                f"| Status | {cache_label} |\n"
                f"{source_info}\n"
            )
        else:
            text = (
                f"## No price found\n\n"
                f"No **{year_month} {pt_name}** data found for **{province}**.\n"
                f"Try another month or check {province} DRC / power exchange website."
            )

        return self._result(text, chart, table)

    # ─── Weather ───

    def _handle_weather(self, entities: dict) -> dict:
        city = entities.get("city") or self.state.current_city
        if not city and self.state.current_province:
            city = get_city_for_province(self.state.current_province)
        if not city:
            return self._result("Which city? e.g. 'Beijing weather'", status="Missing city")

        weather_data = query_weather(city)
        text = format_weather_response(weather_data)
        return self._result(text)

    # ─── Knowledge query (search + LLM summary) ───

    def _handle_knowledge_stream(self, user_input: str, status: str):
        results = web_search(f"new energy {user_input}", max_results=5)

        if not results:
            try:
                messages = [{"role": "system", "content": FINAL_RESPONSE_SYSTEM}, {"role": "user", "content": user_input}]
                text = ""
                for chunk in generate_stream(messages, max_tokens=1024, temperature=0.7):
                    text += chunk
                    yield self._result(text, status=status)
                return
            except Exception as e:
                yield self._result(f"Search and answer unavailable: {e}", status=status)
                return

        search_text = "\n\n".join(
            f"[{i}] {r['title']}\n{r['snippet']}\nSource: {r['url']}"
            for i, r in enumerate(results, 1)
        )
        system_prompt = KNOWLEDGE_QUERY_SYSTEM.replace("{search_results}", search_text)

        try:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            text = ""
            for chunk in generate_stream(messages, max_tokens=1024, temperature=0.7):
                text += chunk
                yield self._result(text, status=status)
            text += "\n\n---\n*Based on web search, for reference only*"
            yield self._result(text, status=status)
        except Exception as e:
            yield self._result(f"## Search results for «{user_input}»\n\n{search_text}", status=status)

    # ─── Chat ───

    def _handle_chat_stream(self, user_input: str, status: str):
        try:
            messages = [{"role": "system", "content": FINAL_RESPONSE_SYSTEM}, {"role": "user", "content": user_input}]
            text = ""
            for chunk in generate_stream(messages, max_tokens=512, temperature=0.8):
                text += chunk
                yield self._result(text, status=status)
        except Exception:
            yield self._result(
                "Hi! I'm a new energy industry assistant.\n\n"
                "I can help with:\n"
                "Feed-in / Desulfurized coal / Commercial & industrial tariffs\n"
                "Weather\n"
                "New energy policy & knowledge\n\n"
                "What can I help you with?",
                status=status
            )

    def reset(self):
        self.state = self.context_mgr.reset()
        self.last_chart = None
        self.last_table = None


# ── Gradio UI ──────────────────────────────────────────────────


def create_ui(agent: NewEnergyAgent) -> gr.Blocks:
    custom_css = """
    .gradio-container { max-width: 1000px !important; margin: auto; }
    .chatbot { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .status-bar { padding: 8px; font-weight: bold; color: #555; background: #f0f0f0; border-radius: 8px; text-align: center; margin-bottom: 10px; }
    """
    # Note: theme and css are passed to launch() in the notebook
    with gr.Blocks(title="New Energy Agent") as demo:

        gr.Markdown(
            "# ⚡ New Energy Industry Assistant\n"
            "Feed-in tariff | Desulfurized coal | Commercial & industrial | Weather | Policy"
        )

        status_bar = gr.Markdown("🟢 Ready | Waiting for input...", elem_classes=["status-bar"])

        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    elem_classes=["chatbot"],
                    height=550,
                    avatar_images=(None, "🤖"),
                    render_markdown=True
                )
            with gr.Column(scale=3):
                with gr.Accordion("📊 Data Visualization & Tables", open=False) as data_accordion:
                    with gr.Tabs():
                        with gr.TabItem("📈 Price Trend"):
                            plot_output = gr.Plot(label="Price Trend")
                        with gr.TabItem("📋 Detailed Data"):
                            table_output = gr.Dataframe(label="Data", wrap=True)

        with gr.Row():
            msg_input = gr.Textbox(placeholder="e.g. Shanghai feed-in tariff / Beijing weather", scale=5, show_label=False)
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Row():
            btn_feedin = gr.Button("Feed-in", size="sm")
            btn_coal = gr.Button("Desulfurized Coal", size="sm")
            btn_com = gr.Button("Commercial", size="sm")
            btn_weather = gr.Button("Weather", size="sm")

        gr.Examples(
            examples=["Shanghai feed-in tariff", "Jiangsu desulfurized coal price", "Beijing weather", "Solar subsidy policy"],
            inputs=msg_input,
        )

        def on_message(message: str, history: list):
            if not message.strip():
                yield history, gr.update(), gr.update(), gr.update(), "Enter a question"
                return
            
            # Universal history format: list of [user_msg, assistant_msg]
            history.append({"role": "user", "content": message.strip()})
            history.append({"role": "assistant", "content": ""})
            
            chart, table = None, None
            status = "Processing..."
            yield history, gr.update(), gr.update(), gr.update(), status
            
            for result in agent.process_stream(message.strip()):
                history[-1]["content"] = result["text"]
                chart = result["chart"]
                table = result["table"]
                status = result.get("status", "")
                
                accordion_upd = gr.update(open=True) if chart or table is not None else gr.update()
                chart_upd = gr.update(value=chart, visible=chart is not None) if chart else gr.update(visible=False)
                table_upd = gr.update(value=table, visible=table is not None) if table is not None else gr.update(visible=False)
                
                yield history, chart_upd, table_upd, accordion_upd, status

        def on_clear():
            agent.reset()
            return [], gr.update(visible=False), gr.update(visible=False), gr.update(open=False), "🟢 Ready | Waiting for input..."

        submit_events = [
            msg_input.submit(on_message, [msg_input, chatbot], [chatbot, plot_output, table_output, data_accordion, status_bar]),
            send_btn.click(on_message, [msg_input, chatbot], [chatbot, plot_output, table_output, data_accordion, status_bar]),
            btn_feedin.click(lambda: "feed-in ", None, msg_input).then(on_message, [msg_input, chatbot], [chatbot, plot_output, table_output, data_accordion, status_bar]),
            btn_coal.click(lambda: "desulfurized coal ", None, msg_input).then(on_message, [msg_input, chatbot], [chatbot, plot_output, table_output, data_accordion, status_bar]),
            btn_com.click(lambda: "commercial ", None, msg_input).then(on_message, [msg_input, chatbot], [chatbot, plot_output, table_output, data_accordion, status_bar]),
            btn_weather.click(lambda: "weather ", None, msg_input).then(on_message, [msg_input, chatbot], [chatbot, plot_output, table_output, data_accordion, status_bar])
        ]
        
        for ev in submit_events:
            ev.then(lambda: "", None, msg_input)

        clear_btn.click(on_clear, None, [chatbot, plot_output, table_output, data_accordion, status_bar])

        gr.Markdown("New Energy Industry Assistant | Public data & wttr.in | AI content for reference only")

    return demo


# ── ngrok ──────────────────────────────────────────────────────


def start_ngrok(port: int = 7860):
    if not NGROK_TOKEN:
        logger.warning("No NGROK_TOKEN, skipping ngrok")
        return None
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(NGROK_TOKEN)
        tunnel = ngrok.connect(port)
        logger.info(f"ngrok: {tunnel.public_url}")
        return tunnel.public_url
    except Exception as e:
        logger.warning(f"ngrok failed: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("New Energy Agent starting...")
    logger.info("=" * 50)

    # Lazy-load model on first request, not here
    ngrok_url = start_ngrok(7860)

    agent = NewEnergyAgent()
    demo = create_ui(agent)

    print()
    print("=" * 50)
    print("  New Energy Agent")
    print("=" * 50)
    if ngrok_url:
        print(f"  ngrok: {ngrok_url}")
    print("  Gradio URL: see cell output for .gradio.live")
    print("=" * 50)

    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        css=".gradio-container{max-width:900px!important}",
        theme=gr.themes.Soft(primary_hue="green"),
    )
