"""
意图路由器 - 基于 Qwen2.5-3B 的意图分类 + 实体抽取
通过 Prompt Engineering (不微调) 实现精确的意图识别和上下文感知
"""

import json
import logging
import re
from typing import TypedDict

from openai import OpenAI

from src.config import VLLM_BASE_URL
from src.context_manager import ContextManager
from src.prompt_templates import build_intent_messages

logger = logging.getLogger(__name__)


class IntentResult(TypedDict):
    intent: str                # price_query | weather_query | knowledge_query | chat | out_of_domain
    entities: dict             # {province, city, month, price_type}
    inherit_from_context: bool
    reasoning: str


class IntentRouter:
    """意图路由: 分类 + 实体抽取"""

    def __init__(self):
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
        self.context_mgr = ContextManager()

    def classify(self, user_input: str, state) -> IntentResult:
        """
        对用户输入进行意图分类和实体抽取

        Args:
            user_input: 用户输入文本
            state: ConversationState 当前对话状态

        Returns:
            IntentResult { intent, entities, inherit_from_context, reasoning }
        """
        # 构建包含上下文的 System Prompt
        context_summary = self.context_mgr.get_context_summary(state)
        messages = build_intent_messages(user_input, context_summary)

        try:
            resp = self.client.chat.completions.create(
                model="qwen",
                messages=messages,
                max_tokens=500,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"意图分类失败: {e}")
            return self._fallback_classify(user_input)

    def _parse_response(self, content: str) -> IntentResult:
        """解析 LLM 返回的 JSON"""
        # 提取 JSON 块
        json_match = re.search(r'```json\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试匹配裸 JSON
            json_match = re.search(r'\{[\s\S]*"intent"[\s\S]*\}', content)
            json_str = json_match.group() if json_match else content

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败，原始输出: {content[:200]}")
            return self._fallback_classify(content)

        return {
            "intent": data.get("intent", "chat"),
            "entities": {
                "province": data.get("entities", {}).get("province"),
                "city": data.get("entities", {}).get("city"),
                "month": data.get("entities", {}).get("month") or "2026-08",
                "price_type": data.get("entities", {}).get("price_type"),
            },
            "inherit_from_context": data.get("inherit_from_context", False),
            "reasoning": data.get("reasoning", ""),
        }

    def _fallback_classify(self, user_input: str) -> IntentResult:
        """关键词兜底分类 (LLM 不可用时的降级方案)"""
        text = user_input.strip()

        # 电价关键词
        price_keywords = {
            "feed_in": ["上网电价", "上网电价", "上网", "标杆电价", "燃煤标杆"],
            "desulfurized_coal": ["脱硫煤", "脱硫煤电价", "燃煤电价", "标煤"],
            "commercial_industrial": ["工商业电价", "工商业", "工商电价", "大工业电价", "一般工商业"],
        }
        for ptype, keywords in price_keywords.items():
            if any(kw in text for kw in keywords):
                return {
                    "intent": "price_query",
                    "entities": self._extract_entities_fallback(text, ptype),
                    "inherit_from_context": self._check_inherit(text),
                    "reasoning": "关键词兜底: 电价查询",
                }

        # 天气关键词
        if any(kw in text for kw in ["天气", "气温", "下雨", "温度", "热不热", "冷不冷", "刮风", "几度"]):
            return {
                "intent": "weather_query",
                "entities": self._extract_entities_fallback(text, None),
                "inherit_from_context": self._check_inherit(text),
                "reasoning": "关键词兜底: 天气查询",
            }

        # 闲聊关键词
        chat_patterns = ["你好", "谢谢", "再见", "你是谁", "嗨", "hello", "hi", "早上好", "晚上好",
                        "晚安", "辛苦了", "在吗", "帮帮我", "你会什么"]
        if any(p in text for p in chat_patterns) or len(text) < 5:
            return {
                "intent": "chat",
                "entities": {},
                "inherit_from_context": False,
                "reasoning": "关键词兜底: 闲聊",
            }

        # 默认按知识查询处理
        return {
            "intent": "knowledge_query",
            "entities": {},
            "inherit_from_context": False,
            "reasoning": "关键词兜底: 默认知识查询",
        }

    def _extract_entities_fallback(self, text: str, price_type: str | None) -> dict:
        """用正则提取省份（兜底方案）"""
        from src.config import PROVINCE_TO_CITY
        province = None
        for p_name in PROVINCE_TO_CITY:
            if p_name in text:
                province = p_name.rstrip("省市自治区")
                break
        # 也检查简称
        if not province:
            short_map = {
                "沪": "上海", "京": "北京", "津": "天津", "渝": "重庆",
                "苏": "江苏", "浙": "浙江", "粤": "广东", "鲁": "山东",
            }
            for short, full in short_map.items():
                if short in text:
                    province = full
                    break

        return {
            "province": province or None,
            "city": None,
            "month": "2026-08",
            "price_type": price_type,
        }

    def _check_inherit(self, text: str) -> bool:
        """检查是否需要上下文继承（关键词）"""
        inherit_indicators = ["呢", "那", "也", "还", "...呢", "呢?", "那...呢"]
        text_clean = text.replace(" ", "")
        return any(ind in text_clean for ind in inherit_indicators)
