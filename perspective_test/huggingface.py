"""
Generic solver module for any Hugging Face text-generation model, run
locally via transformers. Exposes call_huggingface(prompt, model=...) with
the same return shape as gpt.py's call_gpt / gemini.py's call_gemini, so it
can be used as a drop-in solver in main.py for any model on the Hub - just
add its model ID to HUGGINGFACE_MODELS in main.py.
"""
import re
import time

import torch
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList, TextStreamer

# Building a pipeline loads the full model into memory, which is expensive
# - so each model is only built once and reused for every subsequent call,
# keyed by model name.
_PIPELINE_CACHE = {}


def _select_device():
    """Picks the fastest available backend for local inference: Apple
    Silicon GPU (MPS) if present, otherwise CUDA, otherwise plain CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_pipeline(model):
    if model not in _PIPELINE_CACHE:
        print(f"[{model}] Loading model into memory (this only happens once)...")
        # A single explicit device, NOT device_map="auto". auto's memory
        # estimation can misjudge how much room is actually available
        # (seems especially true on Apple Silicon/MPS) and decide to
        # offload some layers to disk instead - that's what "Some
        # parameters are on the meta device because they were offloaded
        # to the disk" means, and it's why generation was crawling: every
        # offloaded layer gets paged in from disk on every forward pass.
        # phi-2 is only ~2.7B params (~5-6GB in fp16), which fits
        # comfortably on one device with no offloading needed at all.
        _PIPELINE_CACHE[model] = pipeline(
            task="text-generation", model=model, model_kwargs={"dtype": "auto"}, device=_select_device()
        )
    return _PIPELINE_CACHE[model]


def _model_supports_chat_template(text_generation_pipeline):
    """Instruction/chat-tuned models (most "-Instruct"/"-Chat" models, e.g.
    DeepSeek-R1) define a chat_template on their tokenizer and expect
    turn-formatted input. Base/completion models (e.g. microsoft/phi-2)
    don't define one, and should just be given the raw prompt string
    instead - passing chat-formatted input to a model with no chat
    template raises an error."""
    return getattr(text_generation_pipeline.tokenizer, "chat_template", None) is not None


class _StopAtNewline(StoppingCriteria):
    """Halts generation as soon as a newline shows up among the newly
    generated tokens. Only attached for base/completion models: those
    never stop on their own and will otherwise burn through the full
    max_new_tokens budget hallucinating fake future turns, even though
    only the first line is ever kept (see the truncation in
    call_huggingface). Chat-template models never get this attached, so
    their normal (possibly multi-line) reasoning is untouched.

    Tracks the most recently generated token at each step (input_ids'
    last position) rather than trying to compute where the prompt ends,
    so there's no risk of it mismatching the pipeline's own tokenization
    of the prompt."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_token_ids = []

    def __call__(self, input_ids, scores, **kwargs):
        self.generated_token_ids.append(input_ids[0, -1].item())
        generated_so_far = self.tokenizer.decode(self.generated_token_ids, skip_special_tokens=True)
        return "\n" in generated_so_far


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


def call_huggingface(prompt, model, max_new_tokens=256, max_retries=3, show_live_output=True):
    """Calls any Hugging Face text-generation model once with the given
    prompt, retrying up to max_retries times if it raises an exception
    (e.g. a transient CUDA or out-of-memory error).

    show_live_output=True prints the model's tokens to the terminal as
    they're generated (via transformers' TextStreamer), so long generations
    show visible progress instead of the terminal sitting silent until the
    whole response is ready. This is purely a side effect for visibility -
    the function's return value is unaffected either way."""
    text_generation_pipeline = _get_pipeline(model)
    tokenizer = text_generation_pipeline.tokenizer
    is_chat_model = _model_supports_chat_template(text_generation_pipeline)

    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            print(f"[{model}] [Attempt {attempt}] Generating...")

            if is_chat_model:
                pipeline_input = [{"role": "user", "content": prompt}]
            else:
                pipeline_input = prompt

            generate_kwargs = {"max_new_tokens": max_new_tokens}
            if show_live_output:
                generate_kwargs["streamer"] = TextStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )

            if not is_chat_model:
                # Stop as soon as a newline shows up, instead of always
                # generating the full max_new_tokens budget. Base models
                # only ever contribute one real line anyway (see the
                # truncation below) - everything past the first newline
                # is guaranteed to be thrown away, so there's no point
                # spending time generating it.
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_StopAtNewline(tokenizer)]
                )

            pipeline_output = text_generation_pipeline(pipeline_input, **generate_kwargs)
            elapsed_seconds = time.time() - start_time

            full_generated_text = _extract_generated_text(pipeline_output, prompt)

            if not is_chat_model:
                # Base/completion models don't stop after answering - they keep
                # completing the document, which for this prompt shape means
                # hallucinating fake future turns (often Advent-of-Code-flavored,
                # given how much GitHub code data these models train on). The
                # real answer is always whatever comes before the first line
                # break; everything after that is hallucinated continuation.
                full_generated_text = full_generated_text.strip().split("\n")[0]

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