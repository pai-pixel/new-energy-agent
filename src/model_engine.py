"""
模型推理引擎 - DeepSeek API (OpenAI 兼容)
封装 OpenAI SDK, 提供统一的生成/JSON生成接口。
所有模块的 LLM 调用都走这里, 便于统一埋点(耗时/错误)。
"""

from __future__ import annotations

import logging                                  # 模块 logger
import time                                      # 记录 API 调用耗时(排查慢响应)
from typing import Optional                      # 类型标注

from openai import OpenAI                        # OpenAI 兼容客户端(DeepSeek 用它)

# 模块级 logger
logger = logging.getLogger(__name__)

# 全局单例客户端(懒加载): 只初始化一次, 复用 TCP 连接提升性能
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """
    获取 DeepSeek API 客户端(单例)。
    首次调用时校验 API Key 并创建; 之后复用全局 _client。
    API Key 缺失时抛 RuntimeError, 提示用户配置。
    """
    global _client                                # 需要修改模块级单例
    from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL  # 惰性导入避免循环依赖
    api_key, base_url, model = LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    if _client is None:                           # 尚未初始化
        if not api_key or api_key == "your-deepseek-api-key-here":  # 占位key视为未配置
            raise RuntimeError(
                "请先设置 DeepSeek API Key!\n"
                "  export DEEPSEEK_API_KEY=sk-xxx\n"
                "或在项目根目录 .env 文件中添加:\n"
                "  DEEPSEEK_API_KEY=sk-xxx"
            )
        _client = OpenAI(base_url=base_url, api_key=api_key)  # 创建客户端
        logger.info(f"[DeepSeek] 客户端已初始化 base_url={base_url} model={model}")
    return _client


def get_model_name() -> str:
    """返回当前使用的模型名(供前端展示/日志)。"""
    from src.config import LLM_MODEL
    return LLM_MODEL


def generate(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    通用生成接口(非流式)。
    - messages: OpenAI 格式的消息列表
    - max_tokens: 生成上限(不超全局 MODEL_MAX_TOKENS)
    - temperature: 采样温度; DeepSeek 不支持 0, 会自动抬到 0.0 以上
    返回模型生成的纯文本; 失败抛异常给调用方处理。
    """
    from src.config import MODEL_MAX_TOKENS       # 全局 token 上限
    client = _get_client()
    model = get_model_name()
    t0 = time.time()                              # 计时开始
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=min(max_tokens, MODEL_MAX_TOKENS),  # 取两者较小值防超限
            temperature=temperature if temperature > 0 else 0.0,  # 兼容不支持0的模型
            stream=False,
        )
        content = resp.choices[0].message.content
        logger.info(
            f"[DeepSeek] 生成完成 耗时{time.time()-t0:.1f}s "
            f"输入{len(messages)}条消息 → {len(content or '')}字"
        )
        return content.strip() if content else ""  # 去首尾空白, None 兜底空串
    except Exception as e:
        # 记录带耗时的错误日志, 便于定位是网络/鉴权/限流
        logger.error(f"DeepSeek API 调用失败 耗时{time.time()-t0:.1f}s: {e}")
        raise                                         # 抛给上层处理


def generate_json(messages: list[dict], max_tokens: int = 500) -> str:
    """
    结构化 JSON 生成(低温度, 让输出更稳定)。
    注意: DeepSeek v4 不支持 temperature=0, 用 0.1 代替以保证确定性。
    用于电价提取等需要机器可读输出的场景。
    """
    return generate(messages, max_tokens=max_tokens, temperature=0.1)


def load_model(model_path: Optional[str] = None):
    """
    兼容性接口: 早期版本支持本地模型加载, 现在纯 API 模式。
    保留签名让旧调用不报错; 实际不做任何加载。
    """
    logger.info("[DeepSeek] API 模式，无需加载本地模型")
    return None, None


def is_loaded() -> bool:
    """检查 API 是否可用(尝试初始化客户端)。"""
    try:
        _get_client()
        return True
    except RuntimeError:
        return False


def get_device() -> str:
    """返回当前推理后端标识(统一为 deepseek-api)。"""
    return "deepseek-api"
