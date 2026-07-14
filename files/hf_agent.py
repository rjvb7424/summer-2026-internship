"""HuggingFace agent: runs a local model, optimised for Apple Silicon (MPS)."""

import gc
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from base_agent import SYSTEM_PROMPT, BaseAgent


def _pick_device():
    """Best available device: MPS on Apple Silicon, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HuggingFaceAgent(BaseAgent):
    """Wraps one local HuggingFace causal LM as a Crafter policy."""

    provider = "huggingface"

    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = _pick_device()
        dtype = getattr(torch, config.TORCH_DTYPE) if self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # device_map pinned to one device streams weights shard-by-shard
        # straight onto MPS -- never holding a full second copy in CPU RAM.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": self.device},
        )
        self.model.eval()
        gc.collect()  # drop loader buffers before the first step
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count())

    # -------------------- generation --------------------

    def _generate(self, user_text):
        prompt = self._apply_template(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
                temperature=config.TEMPERATURE if config.DO_SAMPLE else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _apply_template(self, user_text):
        """Chat-templated prompt, with a plain-text fallback for base models."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # no chat template (e.g. base models like phi-2)
            return f"{SYSTEM_PROMPT}\n\n{user_text} "

    # -------------------- teardown --------------------

    def unload(self):
        """Free model memory before loading the next model."""
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()