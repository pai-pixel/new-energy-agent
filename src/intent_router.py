"""
意图路由器 - 基于 Qwen2.5-3B 的意图分类 + 实体抽取
通过 Prompt Engineering (不微调) 实现精确的意图识别和上下文感知
"""

import json
import logging
import re
from typing import TypedDict

from src.context_manager import ContextManager
from src.prompt_templates import build_intent_messages
from src.model_engine import generate_json

logger = logging.getLogger(__name__)


class IntentResult(TypedDict):
    intent: str
    entities: dict
    inherit_from_context: bool
    reasoning: str


class IntentRouter:
    """意图路由: 分类 + 实体抽取"""

    def __init__(self):
        self.context_mgr = ContextManager()

    def classify(self, user_input: str, state) -> IntentResult:
        context_summary = self.context_mgr.get_context_summary(state)
        messages = build_intent_messages(user_input, context_summary)

        try:
            content = generate_json(messages, max_tokens=500)
            return self._parse_json(content)
        except Exception as e:
            logger.error(f"Intent classify failed: {e}")
            return self._fallback(user_input)

    def _parse_json(self, content: str) -> IntentResult:
        """从 LLM 返回中提取 JSON"""
        # 尝试 ```json ... ``` 块
        m = re.search(r'```json\s*([\s\S]*?)```', content)
        json_str = m.group(1) if m else content
        # 清理可能的前导/尾随非 JSON 文字
        m2 = re.search(r'\{[\s\S]*"intent"[\s\S]*\}', json_str)
        if m2:
            json_str = m2.group()

        data = json.loads(json_str)
        return {
            "intent": data.get("intent", "chat"),
            "entities": {
                "province": (data.get("entities", {}) or {}).get("province"),
                "city": (data.get("entities", {}) or {}).get("city"),
                "month": (data.get("entities", {}) or {}).get("month") or "2026-08",
                "price_type": (data.get("entities", {}) or {}).get("price_type"),
            },
            "inherit_from_context": data.get("inherit_from_context", False),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback(self, user_input: str) -> IntentResult:
        text = user_input.strip()
        price_kw = {
            "feed_in": ["上网电价", "上网", "标杆电价", "燃煤标杆"],
            "desulfurized_coal": ["脱硫煤", "脱硫煤电价", "燃煤电价"],
            "commercial_industrial": ["工商业电价", "工商业", "大工业电价", "一般工商业"],
        }
        for pt, kws in price_kw.items():
            if any(k in text for k in kws):
                return {
                    "intent": "price_query",
                    "entities": self._extract_province(text, pt),
                    "inherit_from_context": self._check_inherit(text),
                    "reasoning": "keyword fallback",
                }
        weather_kw = ["天气", "气温", "下雨", "温度", "热不热", "冷不冷", "几度"]
        if any(k in text for k in weather_kw):
            return {
                "intent": "weather_query",
                "entities": self._extract_province(text, None),
                "inherit_from_context": self._check_inherit(text),
                "reasoning": "keyword fallback",
            }
        chat_kw = ["你好", "谢谢", "再见", "你是谁", "嗨", "hello", "hi", "在吗"]
        if any(k in text for k in chat_kw) or len(text) < 5:
            return {"intent": "chat", "entities": {}, "inherit_from_context": False,
                    "reasoning": "keyword fallback"}
        return {"intent": "knowledge_query", "entities": {}, "inherit_from_context": False,
                "reasoning": "default"}

    def _extract_province(self, text: str, pt: str | None) -> dict:
        from src.config import PROVINCE_TO_CITY
        for p in PROVINCE_TO_CITY:
            if p in text:
                return {"province": p.rstrip("省市自治区"), "city": None,
                        "month": "2026-08", "price_type": pt}
        return {"province": None, "city": None, "month": "2026-08", "price_type": pt}

    def _check_inherit(self, text: str) -> bool:
        return any(i in text.replace(" ", "") for i in ["呢", "那", "也", "还"])
