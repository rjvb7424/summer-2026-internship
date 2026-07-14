"""HuggingFace agent: loads a model, turns observations into Crafter actions."""

import gc
import os
import re

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.75")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from crafter_env import ACTION_NAMES

# ============================================================
# Constants
# ============================================================
SYSTEM_PROMPT = (
    "You are playing Crafter, a 2D survival game on a grid.\n"
    "Valid actions: " + ", ".join(ACTION_NAMES) + ".\n"
    "Rules:\n"
    "- 'do' interacts with the tile you face: chop tree (wood), mine stone, "
    "drink water, attack, eat cow.\n"
    "- Moving toward something faces you at it; you must be adjacent to use 'do'.\n"
    "- Crafting needs a table nearby (place_table costs wood). Iron tools also "
    "need a furnace and coal.\n"
    "- Keep food, drink and energy up: eat, drink, sleep before they hit 0.\n"
    "Reply with exactly one action name and nothing else."
)

# Longest names first so 'move_left' is matched before 'do', etc.
_ACTION_PATTERN = re.compile(
    "|".join(rf"\b{re.escape(name)}\b" for name in sorted(ACTION_NAMES, key=len, reverse=True))
)


# ============================================================
# Device selection
# ============================================================
def _pick_device():
    """Best available device: MPS on Apple Silicon, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Agent
# ============================================================
class HuggingFaceAgent:
    """Wraps one HuggingFace causal LM as a Crafter policy."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = _pick_device()
        dtype = getattr(torch, config.TORCH_DTYPE) if self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count())
        self.history = []  # rolling [(action, outcome), ...]

    # -------------------- policy --------------------

    def choose_action(self, observation_text):
        """Return (action_name, raw_response, valid) for the observation."""
        prompt = self._build_prompt(observation_text)
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
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = _ACTION_PATTERN.search(response)
        if match:
            return match.group(0), response, True
        return "noop", response, False

    def record_outcome(self, action_name, reward):
        """Append the last step to the rolling history shown in prompts."""
        outcome = f"reward {reward:+.1f}" if reward else "no reward"
        self.history.append((action_name, outcome))
        self.history = self.history[-config.HISTORY_LENGTH:]

    def reset_history(self):
        self.history = []

    # -------------------- prompt --------------------

    def _build_prompt(self, observation_text):
        """Chat-templated prompt, with a plain-text fallback for base models."""
        history = (
            "Recent actions:\n"
            + "\n".join(f"- {action} ({outcome})" for action, outcome in self.history)
            if self.history else "Recent actions: none yet"
        )
        user_text = f"{observation_text}\n\n{history}\n\nNext action:"
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