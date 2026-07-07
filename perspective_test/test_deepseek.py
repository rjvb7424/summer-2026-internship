"""
Test suite for deepseek.py.

None of these tests download or run the real DeepSeek model - that needs
network access to Hugging Face and (for a model the size of DeepSeek-R1)
substantial GPU hardware, neither of which is available in this sandbox.
Instead, _get_pipeline is monkeypatched with a small fake pipeline object
that mimics transformers' real output shape, so the surrounding logic
(retries, <think> tag splitting, token counting, the dict returned to the
caller) can be verified in isolation. Point this module at a real model in
your own environment and the same call_deepseek function will work
unchanged.
"""
import pytest

import deepseek


class FakeTokenizer:
    """Stand-in for a real Hugging Face tokenizer: 'encodes' text as a list
    of whitespace-split tokens, which is all these tests need in order to
    check that prompt/answer/thinking lengths get counted at all - the
    exact tokenization scheme doesn't matter for testing call_deepseek's
    logic."""

    def encode(self, text):
        return text.split()


class FakePipeline:
    """Stand-in for a transformers text-generation pipeline. Call it like
    the real pipeline (a list of chat messages in, a chat-shaped
    conversation out) but return whatever canned text was configured,
    optionally raising an exception on the first N calls to test the
    retry loop."""

    def __init__(self, reply_text, raise_for_first_n_calls=0):
        self.tokenizer = FakeTokenizer()
        self.reply_text = reply_text
        self.raise_for_first_n_calls = raise_for_first_n_calls
        self.call_count = 0

    def __call__(self, conversation, max_new_tokens):
        self.call_count += 1
        if self.call_count <= self.raise_for_first_n_calls:
            raise RuntimeError("simulated transient failure")
        return [{"generated_text": conversation + [{"role": "assistant", "content": self.reply_text}]}]


@pytest.fixture(autouse=True)
def clear_pipeline_cache():
    """Each test gets a clean cache, since call_deepseek reuses whatever
    pipeline is already cached for a given model name."""
    deepseek._PIPELINE_CACHE.clear()
    yield
    deepseek._PIPELINE_CACHE.clear()


def install_fake_pipeline(monkeypatch, fake_pipeline, model="fake-model"):
    monkeypatch.setattr(deepseek, "_get_pipeline", lambda requested_model: fake_pipeline)


# ----------------------------------------------------------------------
# Splitting <think> reasoning from the final answer
# ----------------------------------------------------------------------


def test_split_thinking_and_answer_separates_think_tags():
    generated = "<think>I should turn right first</think>RIGHT"
    thinking, answer = deepseek._split_thinking_and_answer(generated)
    assert thinking == "I should turn right first"
    assert answer == "RIGHT"


def test_split_thinking_and_answer_handles_missing_think_tags():
    thinking, answer = deepseek._split_thinking_and_answer("just FORWARD, no reasoning shown")
    assert thinking == ""
    assert answer == "just FORWARD, no reasoning shown"


# ----------------------------------------------------------------------
# call_deepseek: happy path
# ----------------------------------------------------------------------


def test_call_deepseek_returns_the_answer_text_and_token_counts(monkeypatch):
    fake_pipeline = FakePipeline("<think>a few words of thinking</think>FORWARD")
    install_fake_pipeline(monkeypatch, fake_pipeline)

    result = deepseek.call_deepseek("what should I do next", model="fake-model")

    assert result["text"] == "FORWARD"
    assert result["thinking_tokens"] == 5  # "a few words of thinking"
    assert result["output_tokens"] == 1  # "FORWARD"
    assert result["prompt_tokens"] == 5  # "what should I do next"
    assert result["total_tokens"] == result["prompt_tokens"] + result["thinking_tokens"] + result["output_tokens"]
    assert result["model_version"] == "fake-model"
    assert result["is_partial"] is False


def test_call_deepseek_handles_a_response_with_no_thinking_tags(monkeypatch):
    fake_pipeline = FakePipeline("FORWARD")
    install_fake_pipeline(monkeypatch, fake_pipeline)

    result = deepseek.call_deepseek("move now", model="fake-model")

    assert result["text"] == "FORWARD"
    assert result["thinking_tokens"] == 0


# ----------------------------------------------------------------------
# call_deepseek: retries and failure
# ----------------------------------------------------------------------


def test_call_deepseek_retries_after_a_transient_error(monkeypatch):
    fake_pipeline = FakePipeline("RIGHT", raise_for_first_n_calls=2)
    install_fake_pipeline(monkeypatch, fake_pipeline)
    monkeypatch.setattr(deepseek.time, "sleep", lambda seconds: None)  # skip real backoff delay

    result = deepseek.call_deepseek("move now", model="fake-model", max_retries=3)

    assert result is not None
    assert result["text"] == "RIGHT"
    assert fake_pipeline.call_count == 3


def test_call_deepseek_gives_up_after_max_retries(monkeypatch):
    fake_pipeline = FakePipeline("RIGHT", raise_for_first_n_calls=99)
    install_fake_pipeline(monkeypatch, fake_pipeline)
    monkeypatch.setattr(deepseek.time, "sleep", lambda seconds: None)

    result = deepseek.call_deepseek("move now", model="fake-model", max_retries=2)

    assert result is None
    assert fake_pipeline.call_count == 2


# ----------------------------------------------------------------------
# The pipeline cache is reused across calls
# ----------------------------------------------------------------------


def test_pipeline_is_only_constructed_once_per_model(monkeypatch):
    build_count = 0

    def fake_pipeline_constructor(task, model):
        nonlocal build_count
        build_count += 1
        return FakePipeline("FORWARD")

    monkeypatch.setattr(deepseek, "pipeline", fake_pipeline_constructor)

    deepseek.call_deepseek("first call", model="fake-model")
    deepseek.call_deepseek("second call", model="fake-model")

    assert build_count == 1


# ----------------------------------------------------------------------
# Handles the plain-string (non chat-template) pipeline output shape too
# ----------------------------------------------------------------------


def test_extract_generated_text_handles_plain_string_output_with_prompt_echoed():
    pipeline_output = [{"generated_text": "original prompt and then the reply"}]
    result = deepseek._extract_generated_text(pipeline_output, prompt="original prompt and then ")
    assert result == "the reply"


def test_extract_generated_text_handles_chat_message_list_output():
    pipeline_output = [{"generated_text": [
        {"role": "user", "content": "original prompt"},
        {"role": "assistant", "content": "the reply"},
    ]}]
    result = deepseek._extract_generated_text(pipeline_output, prompt="original prompt")
    assert result == "the reply"