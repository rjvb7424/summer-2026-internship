import time
from threading import Thread
from transformers import pipeline, TextIteratorStreamer

_PIPELINE_CACHE = {}

def get_pipeline(model):
    """Load and cache a Hugging Face text-generation pipeline."""
    if model not in _PIPELINE_CACHE:
        print(f"[{model}] Loading model into memory...")

        pipe = pipeline(
            task="text-generation",
            model=model,
            dtype="auto",
            device_map="auto",
            model_kwargs={
                "max_memory": {"mps": "10GiB", "cpu": "2GiB"},
            },
        )
    pipe.generation_config.max_new_tokens = 1024
    pipe.generation_config.max_length = None
    pipe.tokenizer.clean_up_tokenization_spaces = False
    _PIPELINE_CACHE[model] = pipe

    return _PIPELINE_CACHE[model]

def call_huggingface(prompt, model, max_retries=3):
    """Call a Hugging Face chat model, retrying up to max_retries times."""
    text_generation_pipeline = get_pipeline(model)
    tokenizer = text_generation_pipeline.tokenizer
    messages = [{"role": "user", "content": prompt}]

    prompt_tokens = len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    for attempt in range(1, max_retries + 1):
        partial_text = ""
        generation_error = []
        start = time.time()

        try:
            print(f"[{model}] [Attempt {attempt}] Streaming request...")

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            def generate():
                try:
                    text_generation_pipeline(
                        messages,
                        streamer=streamer,
                        return_full_text=False,
                    )
                except Exception as error:
                    generation_error.append(error)
                    streamer.on_finalized_text("", stream_end=True)

            thread = Thread(target=generate)
            thread.start()

            # Consume the stream to collect the output (no console echo).
            for chunk in streamer:
                partial_text += chunk

            thread.join()
            elapsed = time.time() - start

            if generation_error:
                raise generation_error[0]

            print(
                f"[{model}] [Attempt {attempt}] "
                f"Stream finished after {elapsed:.1f}s"
            )

            if not partial_text:
                print(f"[{model}] No text returned. Skipping.")
                return None

            output_tokens = len(
                tokenizer.encode(
                    partial_text,
                    add_special_tokens=False,
                )
            )

            return {
                "text": partial_text,
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": None,
                "total_tokens": prompt_tokens + output_tokens,
                "finish_reason": None,
                "model_version": model,
                "is_partial": False,
            }

        except Exception as error:
            elapsed = time.time() - start

            if partial_text:
                output_tokens = len(
                    tokenizer.encode(
                        partial_text,
                        add_special_tokens=False,
                    )
                )

                print(
                    f"[{model}] [Attempt {attempt}] Generation failed after "
                    f"{elapsed:.1f}s, but returned {len(partial_text)} "
                    "characters. Returning partial result."
                )

                return {
                    "text": partial_text,
                    "elapsed_seconds": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": None,
                    "total_tokens": prompt_tokens + output_tokens,
                    "finish_reason": None,
                    "model_version": model,
                    "is_partial": True,
                }

            wait = 2 ** attempt
            print(
                f"[{model}] [Attempt {attempt}] Error: {error}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    print(f"[{model}] Max retries exceeded. Giving up on this model.")
    return None
