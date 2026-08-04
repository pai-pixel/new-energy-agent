"""
模型推理引擎 - transformers + autoawq 直接加载推理
无需 vLLM，兼容 Colab T4 环境: numpy 1.26, torch 2.3, CUDA 12.1
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from pathlib import Path

logger = logging.getLogger(__name__)

# 全局单例
_model = None
_tokenizer = None
_device = None


def load_model(model_path: str | None = None):
    """加载 Qwen2.5-3B-Instruct-AWQ 模型 (仅一次)"""
    global _model, _tokenizer, _device

    if _model is not None:
        logger.info("Model already loaded")
        return _model, _tokenizer

    model_id = model_path or "Qwen/Qwen2.5-3B-Instruct-AWQ"

    logger.info(f"Loading model: {model_id}")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {_device}")

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return _model, _tokenizer


def generate(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """
    推理接口 - ChatML 格式

    Args:
        messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        max_tokens: 最大生成 tokens
        temperature: 温度
        stream: 是否流式 (暂不支持)

    Returns:
        生成的文本
    """
    global _model, _tokenizer

    if _model is None:
        load_model()

    # 用 tokenizer 的 chat template
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.1,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
        )

    # 去掉输入部分
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    response = _tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def generate_json(
    messages: list[dict],
    max_tokens: int = 500,
) -> str:
    """低温度推理，用于结构化 JSON 输出"""
    return generate(messages, max_tokens=max_tokens, temperature=0.0)


def is_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    return _device or "unknown"
