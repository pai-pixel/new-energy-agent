"""
模型推理引擎 - DeepSeek API (OpenAI 兼容)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    api_key, base_url, model = LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    if _client is None:
        if not api_key or api_key == "your-deepseek-api-key-here":
            raise RuntimeError(
                "请先设置 DeepSeek API Key!\n"
                "  export DEEPSEEK_API_KEY=sk-xxx\n"
                "或在项目根目录 .env 文件中添加:\n"
                "  DEEPSEEK_API_KEY=sk-xxx"
            )
        _client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"[DeepSeek] 客户端已初始化 base_url={base_url} model={model}")
    return _client


def get_model_name() -> str:
    from src.config import LLM_MODEL
    return LLM_MODEL


def generate(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    from src.config import MODEL_MAX_TOKENS
    client = _get_client()
    model = get_model_name()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=min(max_tokens, MODEL_MAX_TOKENS),
            temperature=temperature if temperature > 0 else 0.0,
            stream=False,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {e}")
        raise


def generate_json(messages: list[dict], max_tokens: int = 500) -> str:
    """低温度推理 (结构化 JSON 输出)。注意：DeepSeek v4 不支持 temperature=0，用 0.1 代替"""
    return generate(messages, max_tokens=max_tokens, temperature=0.1)


def load_model(model_path: Optional[str] = None):
    """DeepSeek API 模式下无需加载本地模型，此函数保留接口兼容性"""
    logger.info("[DeepSeek] API 模式，无需加载本地模型")
    return None, None


def is_loaded() -> bool:
    """检查 API 是否可用"""
    try:
        _get_client()
        return True
    except RuntimeError:
        return False


def get_device() -> str:
    """返回当前推理后端"""
    return "deepseek-api"
