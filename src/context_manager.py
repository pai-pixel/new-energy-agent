"""
上下文管理器 - 多轮对话状态跟踪与参数继承
支持: 省份继承、电价类型继承、查询类型场景切换
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationState:
    """对话状态 - 跟踪当前会话上下文"""
    current_province: Optional[str] = None       # 当前省份（如 "上海"）
    current_city: Optional[str] = None           # 当前城市（如 "上海"）
    current_price_type: Optional[str] = None     # feed_in | desulfurized_coal | commercial_industrial
    current_query_type: Optional[str] = None     # price_query | weather_query | knowledge_query | chat
    round_count: int = 0                      # 会话轮数
    history: list[dict] = field(default_factory=list)  # 最近10轮对话摘要


class ContextManager:
    """管理对话上下文的继承与更新"""

    MAX_HISTORY = 10

    def update(self, state: ConversationState, intent: str, entities: dict,
               inherit: bool = False) -> ConversationState:
        """
        根据新意图和实体更新对话状态

        Args:
            state: 当前状态
            intent: 意图分类结果
            entities: 实体抽取结果
            inherit: LLM 判断是否需要继承

        Returns:
            更新后的状态
        """
        state.round_count += 1

        # 如果不在上下文中继承，则重置关联字段
        if not inherit:
            # 全量更新
            if entities.get("province"):
                state.current_province = entities["province"]
                # 省份变化时同步更新城市映射
                from src.config import get_city_for_province
                state.current_city = get_city_for_province(entities["province"])

            if entities.get("city"):
                state.current_city = entities["city"]

            if entities.get("price_type"):
                state.current_price_type = entities["price_type"]

        else:
            # 增量继承: 只更新用户明确提到的字段
            if entities.get("province"):
                state.current_province = entities["province"]
                from src.config import get_city_for_province
                state.current_city = get_city_for_province(entities["province"])

            if entities.get("city"):
                state.current_city = entities["city"]

            if entities.get("price_type"):
                state.current_price_type = entities["price_type"]
            # 如果 price_type 为 null，保持上一轮的值

        # 更新查询类型
        state.current_query_type = intent

        # 记录历史摘要
        state.history.append({
            "round": state.round_count,
            "intent": intent,
            "province": state.current_province,
            "price_type": state.current_price_type,
        })
        # 只保留最近 N 轮
        if len(state.history) > self.MAX_HISTORY:
            state.history = state.history[-self.MAX_HISTORY:]

        return state

    def get_context_summary(self, state: ConversationState) -> str:
        """生成上下文摘要，注入到 System Prompt 中"""
        if state.round_count == 0:
            return "（新对话，无上下文）"

        parts = []
        if state.current_province:
            parts.append(f"- 当前省份: {state.current_province}")
        if state.current_city:
            parts.append(f"- 当前城市: {state.current_city}")
        if state.current_price_type:
            from src.config import PRICE_TYPE_MAP
            name = PRICE_TYPE_MAP.get(state.current_price_type, state.current_price_type)
            parts.append(f"- 当前电价类型: {name}")
        if state.current_query_type:
            parts.append(f"- 当前查询类型: {state.current_query_type}")

        if state.history:
            parts.append(f"- 最近对话 ({len(state.history)} 轮):")
            for h in state.history[-3:]:
                parts.append(f"  第{h['round']}轮: {h['intent']} province={h['province']} price_type={h['price_type']}")

        return "\n".join(parts) if parts else "（无上下文）"

    def get_status_text(self, state: ConversationState) -> str:
        """生成 Gradio 状态栏文本"""
        from src.config import PRICE_TYPE_MAP

        province = state.current_province or "未指定"
        price_type = PRICE_TYPE_MAP.get(state.current_price_type, "") if state.current_price_type else ""
        return f"📍 {province} | {price_type} | 第 {state.round_count} 轮"

    def reset(self) -> ConversationState:
        """重置对话状态"""
        return ConversationState()
