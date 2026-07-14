from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


class HuggingFaceAgent:
    """Loads one Hugging Face chat model and returns one Crafter action."""

    def __init__(self):
        self.device = self._resolve_device(config.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME
        )

        dtype = self._dtype_for_device(self.device)

        print(
            f"[model] Loading {config.MODEL_NAME} "
            f"on {self.device} with dtype={dtype}..."
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _dtype_for_device(device: str):
        if device == "mps":
            return torch.float16
        if device == "cuda":
            return torch.float16
        return torch.float32

    def choose_action(self, observation: str) -> tuple[str, str]:
        system_message = (
            "You control an agent in a small grid world. "
            "Plan briefly, but your final answer must contain exactly one "
            "valid action name and no other action names."
        )

        user_message = observation + (
            "\nRespond in this exact format:\n"
            "ACTION: <one valid action>"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        inputs = self._prepare_inputs(messages)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
                temperature=(
                    config.TEMPERATURE
                    if config.DO_SAMPLE
                    else None
                ),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = output[0, prompt_length:]
        raw_response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        action = self._parse_action(raw_response)
        return action, raw_response

    def _prepare_inputs(self, messages):
        if self.tokenizer.chat_template:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            plain_prompt = (
                f"System: {messages[0]['content']}\n"
                f"User: {messages[1]['content']}\n"
                "Assistant:"
            )
            encoded = self.tokenizer(
                plain_prompt,
                return_tensors="pt",
            )

        return {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }

    def _parse_action(self, response: str) -> str:
        lowered = response.lower()

        explicit = re.search(
            r"action\s*:\s*([a-z_]+)",
            lowered,
        )
        if explicit:
            candidate = explicit.group(1)
            if candidate in config.VALID_ACTIONS:
                return candidate

        # Fallback: choose the first valid action mentioned.
        matches = []
        for action in config.VALID_ACTIONS:
            match = re.search(
                rf"\b{re.escape(action)}\b",
                lowered,
            )
            if match:
                matches.append((match.start(), action))

        if matches:
            matches.sort()
            return matches[0][1]

        print(
            f"[warning] Could not parse model response: "
            f"{response!r}. Using noop."
        )
        return "noop"
