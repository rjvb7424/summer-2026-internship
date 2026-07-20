"""
models/huggingface_api.py
=========================

Runs Hugging Face models in the cloud via the HF Inference Providers router,
which exposes an OpenAI-compatible chat-completions endpoint. This is its own
class so HF cloud is configured, logged, and debugged separately from OpenAI
proper - even though it reuses the same battle-tested request/retry transport.

Config:
    backend: huggingface-api      (aliases: hf-api, hf-cloud)
    name:    <org>/<model>        e.g. Qwen/Qwen2.5-7B-Instruct
Key:
    HF_TOKEN in your .env (token with "Make calls to Inference Providers").
"""

from __future__ import annotations

from models.openai_api import OpenAIModel

HF_ROUTER_URL = "https://router.huggingface.co/v1"
HF_KEY_ENV = "HF_TOKEN"


class HuggingFaceAPIModel(OpenAIModel):
    """A HuggingFace Inference Providers model (OpenAI-compatible router)."""

    def __init__(
        self,
        name: str,
        api_key_env: str = HF_KEY_ENV,
        base_url: str = HF_ROUTER_URL,
        **kwargs,
    ):
        super().__init__(
            name,
            api_key_env=api_key_env,
            base_url=base_url,
            **kwargs,
        )