"""
模型推理引擎 - 双后端自适应
- transformers:  直接加载模型到内存 (兼容性最好，~25 tok/s)
- vllm:          通过 OpenAI-compatible API (速度快，~40 tok/s)

通过环境变量 INFERENCE_BACKEND 控制:
  export INFERENCE_BACKEND=vllm        # vLLM API 模式
  export INFERENCE_BACKEND=transformers # 直接加载模式 (默认)
"""

import logging
import os

logger = logging.getLogger(__name__)

VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1")

# ─── transformers 后端 (直接加载) ─────────────────────────────────

_tf_model = None
_tf_tokenizer = None
_tf_device = None


def _load_model_transformers(model_path: str | None = None):
    """直接加载 AWQ 量化模型到 GPU"""
    global _tf_model, _tf_tokenizer, _tf_device
    if _tf_model is not None:
        return _tf_model, _tf_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_path or "Qwen/Qwen2.5-3B-Instruct-AWQ"
    logger.info(f"[transformers] Loading: {model_id}")

    _tf_device = "cuda" if torch.cuda.is_available() else "cpu"

    _tf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _tf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    logger.info(f"[transformers] Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return _tf_model, _tf_tokenizer


def _generate_transformers(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    import torch
    global _tf_model, _tf_tokenizer, _tf_device
    if _tf_model is None:
        _load_model_transformers()

    text = _tf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tf_tokenizer(text, return_tensors="pt").to(_tf_device)

    with torch.no_grad():
        outputs = _tf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.1,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=_tf_tokenizer.pad_token_id or _tf_tokenizer.eos_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    return _tf_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _generate_transformers_stream(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
    import torch
    from threading import Thread
    from transformers import TextIteratorStreamer
    global _tf_model, _tf_tokenizer, _tf_device
    if _tf_model is None:
        _load_model_transformers()

    text = _tf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tf_tokenizer(text, return_tensors="pt").to(_tf_device)

    streamer = TextIteratorStreamer(_tf_tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10.0)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0.1,
        do_sample=temperature > 0,
        top_p=0.9,
        pad_token_id=_tf_tokenizer.pad_token_id or _tf_tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=_tf_model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


# ─── vLLM 后端 (OpenAI-compatible API) ────────────────────────────

_vllm_client = None


def _get_vllm_client():
    global _vllm_client
    if _vllm_client is None:
        from openai import OpenAI
        _vllm_client = OpenAI(base_url=VLLM_API_URL, api_key="vllm")
    return _vllm_client


def _generate_vllm(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    client = _get_vllm_client()
    resp = client.chat.completions.create(
        model="qwen",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _generate_vllm_stream(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
    client = _get_vllm_client()
    stream = client.chat.completions.create(
        model="qwen",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


# ─── 统一接口 ─────────────────────────────────────────────────────


def load_model(model_path: str | None = None):
    """加载模型 (vLLM 模式下无需调用， transformers 模式下预加载)"""
    if os.environ.get("INFERENCE_BACKEND", "vllm") == "vllm":
        logger.info("[vLLM] Model managed by vLLM server, skip local loading")
        return None, None
    return _load_model_transformers(model_path)


def generate(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """统一推理接口"""
    if os.environ.get("INFERENCE_BACKEND", "vllm") == "vllm":
        return _generate_vllm(messages, max_tokens, temperature)
    return _generate_transformers(messages, max_tokens, temperature)


def generate_stream(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7):
    """统一推理流式接口"""
    if os.environ.get("INFERENCE_BACKEND", "vllm") == "vllm":
        yield from _generate_vllm_stream(messages, max_tokens, temperature)
    else:
        yield from _generate_transformers_stream(messages, max_tokens, temperature)


def generate_json(messages: list[dict], max_tokens: int = 500) -> str:
    """低温度推理 (结构化 JSON)"""
    return generate(messages, max_tokens=max_tokens, temperature=0.0)


def is_loaded() -> bool:
    if os.environ.get("INFERENCE_BACKEND", "vllm") == "vllm":
        try:
            _get_vllm_client().models.list()
            return True
        except Exception:
            return False
    return _tf_model is not None


def get_device() -> str:
    return "cuda:vllm" if os.environ.get("INFERENCE_BACKEND", "vllm") == "vllm" else (_tf_device or "unknown")
