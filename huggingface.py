import gc
import time

import torch
from transformers import pipeline

from config import MAX_MEMORY, MAX_NEW_TOKENS

_PIPELINE_CACHE = {}

def get_pipeline(model):
    """Load and cache a Hugging Face text-generation pipeline."""
    if model not in _PIPELINE_CACHE:
        print(f"[{model}] Loading model into memory...")

        text_generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            dtype="auto",
            device_map="auto",
            model_kwargs={"max_memory": MAX_MEMORY},
        )
        text_generation_pipeline.generation_config.max_new_tokens = MAX_NEW_TOKENS
        text_generation_pipeline.generation_config.max_length = None
        text_generation_pipeline.tokenizer.clean_up_tokenization_spaces = False

        _PIPELINE_CACHE[model] = text_generation_pipeline

    return _PIPELINE_CACHE[model]

def unload_models():
    """Free cached pipelines and MPS memory before loading the next model."""
    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def build_messages(prompt, system_prompt=None, history=None):
    """Build a chat message list from the prompt, optional system prompt, and history."""
    messages = list(history) if history else []

    if system_prompt and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})
    return messages

def build_result(text, tokenizer, elapsed, prompt_tokens, model):
    """Pack generated text and token counts into the standard result dict."""
    output_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    return {
        "text": text,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": None,
        "total_tokens": prompt_tokens + output_tokens,
        "finish_reason": None,
        "model_version": model,
        "is_partial": False,
    }

def call_huggingface(prompt, model, system_prompt=None, history=None, max_retries=3):
    """Run a single generation request against a local Hugging Face model."""
    text_generation_pipeline = get_pipeline(model)
    tokenizer = text_generation_pipeline.tokenizer
    messages = build_messages(prompt, system_prompt, history)

    prompt_tokens = len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    for attempt in range(1, max_retries + 1):
        start = time.time()

        try:
            print(f"[{model}] [Attempt {attempt}] Generating...")

            outputs = text_generation_pipeline(messages, return_full_text=False)
            generated_text = outputs[0]["generated_text"]

            elapsed = time.time() - start
            print(f"[{model}] [Attempt {attempt}] Finished after {elapsed:.1f}s")

            if not generated_text:
                print(f"[{model}] No text returned. Skipping.")
                return None

            return build_result(generated_text, tokenizer, elapsed, prompt_tokens, model)

        except Exception as error:
            wait = 2 ** attempt
            print(
                f"[{model}] [Attempt {attempt}] Error: {error}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    print(f"[{model}] Max retries exceeded. Giving up on this model.")
    return None