"""
models/huggingface_local.py
===========================

Runs a HuggingFace model locally with ``transformers``. Built for an Apple
Silicon (M-series) laptop: it auto-selects the MPS backend, loads lazily, and
frees memory on ``unload()`` so only one model sits in RAM at a time.

``torch``/``transformers`` are imported lazily inside the methods, so importing
this module (and running mock-only experiments) never requires them.

AI-facing: ``generate`` performs no printing or input.
"""

from __future__ import annotations

import gc
import logging
import time

from models.base import LanguageModel
from models.conversation import ConversationMemory

LOG = logging.getLogger("crafter_experiment.models.hf")


class HuggingFaceModel(LanguageModel):
    """A local causal-LM agent loaded from the HuggingFace Hub."""

    def __init__(
        self,
        name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        dtype: str = "auto",
        device: str = "auto",
        token_env: str | None = None,
        history_turns: int = 0,
    ):
        super().__init__(name)
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._dtype = dtype
        self._device_pref = device
        self._token_env = token_env  # env var holding an HF token (for gated repos)
        self._memory = ConversationMemory(history_turns)
        self._model = None
        self._tokenizer = None
        self._device = None

    # -- lifecycle ------------------------------------------------------------
    def load(self) -> None:
        import os

        if "/" not in self.name and not os.path.isdir(self.name):
            raise RuntimeError(
                f"'{self.name}' is not a valid HuggingFace model id - these are "
                f"'org/name', e.g. 'microsoft/Phi-4-mini-instruct'. "
                f"Add the org prefix in your config."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._resolve_device(torch)
        torch_dtype = self._resolve_dtype(torch)
        LOG.info("Loading %s on %s (%s)...", self.name, self._device, torch_dtype)

        # token=False means "send no token" - public models download anonymously
        # and any broken/expired token cached on the machine is ignored. For a
        # gated repo, set `hf_token_env: HF_TOKEN` in the model's config and put
        # a valid token in that environment variable.
        token = os.environ.get(self._token_env) if self._token_env else False

        self._tokenizer = AutoTokenizer.from_pretrained(self.name, token=token)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=torch_dtype, token=token
        ).to(self._device)
        self._model.eval()

    def unload(self) -> None:
        import torch

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # -- inference ------------------------------------------------------------
    def reset(self) -> None:
        self._memory.reset()

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        import torch

        # Include recent turns so the model remembers its own moves.
        messages = self._memory.messages(system_prompt, user_prompt)
        enc = self._encode(messages).to(self._device)
        prompt_len = enc["input_ids"].shape[-1]
        do_sample = self._temperature > 0.0

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **enc,
                max_new_tokens=self._max_new_tokens,
                do_sample=do_sample,
                temperature=self._temperature if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start

        new_tokens = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.last_usage = int(prompt_len) + int(new_tokens.shape[-1])
        self._memory.record(user_prompt, text)
        return text, elapsed

    # -- internals ------------------------------------------------------------
    def _encode(self, messages: list[dict]):
        """Encode a list of chat messages into a BatchEncoding. Uses the chat
        template if present, else flattens to role-labelled text."""
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages).strip()
        return tok(text, return_tensors="pt")

    def _resolve_device(self, torch) -> str:
        pref = self._device_pref
        if pref != "auto":
            return pref
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, torch):
        if self._dtype == "auto":
            return "auto"
        return getattr(torch, self._dtype)
