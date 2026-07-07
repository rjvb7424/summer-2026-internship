"""
Generic solver module for any Hugging Face text-generation model, run
locally via transformers. Exposes call_huggingface(prompt, model=...) with
the same return shape as gpt.py's call_gpt / gemini.py's call_gemini, so it
can be used as a drop-in solver in main.py for any model on the Hub - just
add its model ID to HUGGINGFACE_MODELS in main.py.
"""
import re
import time

from transformers import pipeline

# Building a pipeline loads the full model into memory, which is expensive
# - so each model is only built once and reused for every subsequent call,
# keyed by model name.
_PIPELINE_CACHE = {}


def _get_pipeline(model):
    if model not in _PIPELINE_CACHE:
        print(f"[{model}] Loading model into memory (this only happens once)...")
        _PIPELINE_CACHE[model] = pipeline(task="text-generation", model=model)
    return _PIPELINE_CACHE[model]


def _model_supports_chat_template(text_generation_pipeline):
    """Instruction/chat-tuned models (most "-Instruct"/"-Chat" models, e.g.
    DeepSeek-R1) define a chat_template on their tokenizer and expect
    turn-formatted input. Base/completion models (e.g. microsoft/phi-2)
    don't define one, and should just be given the raw prompt string
    instead - passing chat-formatted input to a model with no chat
    template raises an error."""
    return getattr(text_generation_pipeline.tokenizer, "chat_template", None) is not None


def _split_thinking_and_answer(generated_text):
    """Some reasoning models (e.g. DeepSeek-R1) wrap chain-of-thought
    reasoning in <think>...</think> before the actual answer. Splits the
    two apart so they can be reported (and token-counted) separately.
    Models that don't use this convention (e.g. phi-2) just get an empty
    thinking_text and the whole response as the answer."""
    match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
    if not match:
        return "", generated_text.strip()
    thinking_text = match.group(1).strip()
    answer_text = generated_text[match.end():].strip()
    return thinking_text, answer_text


def _extract_generated_text(pipeline_output, prompt):
    """Handles both possible shapes of transformers text-generation output:
    - Chat-template models return generated_text as the full conversation
      list; the reply is the last message.
    - Plain completion models (like phi-2) return generated_text as a
      single string containing the original prompt followed by the
      continuation.
    """
    generated_field = pipeline_output[0]["generated_text"]
    if isinstance(generated_field, list):
        return generated_field[-1]["content"]
    if generated_field.startswith(prompt):
        return generated_field[len(prompt):]
    return generated_field


def call_huggingface(prompt, model, max_new_tokens=1024, max_retries=3):
    """Calls any Hugging Face text-generation model once with the given
    prompt, retrying up to max_retries times if it raises an exception
    (e.g. a transient CUDA or out-of-memory error)."""
    text_generation_pipeline = _get_pipeline(model)
    tokenizer = text_generation_pipeline.tokenizer

    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            print(f"[{model}] [Attempt {attempt}] Generating...")

            if _model_supports_chat_template(text_generation_pipeline):
                pipeline_input = [{"role": "user", "content": prompt}]
            else:
                pipeline_input = prompt

            pipeline_output = text_generation_pipeline(pipeline_input, max_new_tokens=max_new_tokens)
            elapsed_seconds = time.time() - start_time

            full_generated_text = _extract_generated_text(pipeline_output, prompt)
            thinking_text, answer_text = _split_thinking_and_answer(full_generated_text)

            prompt_token_count = len(tokenizer.encode(prompt))
            thinking_token_count = len(tokenizer.encode(thinking_text)) if thinking_text else 0
            answer_token_count = len(tokenizer.encode(answer_text)) if answer_text else 0
            output_token_count = thinking_token_count + answer_token_count

            print(f"[{model}] [Attempt {attempt}] Done after {elapsed_seconds:.1f}s")

            return {
                "text": answer_text,
                "elapsed_seconds": elapsed_seconds,
                "prompt_tokens": prompt_token_count,
                "output_tokens": answer_token_count,
                "thinking_tokens": thinking_token_count,
                "total_tokens": prompt_token_count + output_token_count,
                "model_version": model,
                "is_partial": False,
            }

        except Exception as e:
            wait_seconds = 2 ** attempt
            print(f"[{model}] [Attempt {attempt}] Error: {e}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    print(f"[{model}] Max retries exceeded. Giving up.")
    return None