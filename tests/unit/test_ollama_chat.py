"""The Ollama chat adapter's requests and error handling, against a stub server (no Ollama)."""

import json
import re

import httpx
import pytest

from rag.adapters.ollama_chat import OllamaChat
from rag.domain.errors import (
    AnswerTooLongError,
    GenerationError,
    ModelUnavailableError,
    PromptTooLongError,
)
from rag.domain.models import Completion

TOO_LONG_0_34 = json.dumps(
    {
        "error": json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": "request (6046 tokens) exceeds the available context size "
                    "(4096 tokens), try increasing it",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": 6046,
                    "n_ctx": 4096,
                }
            }
        )
    }
)
"""How Ollama 0.34 refuses a prompt that does not fit, with truncation turned off."""
TOO_LONG_0_12 = json.dumps({"error": "the input length exceeds the context length"})
"""The same refusal from Ollama 0.12.6."""


def reply(
    content: str = "Five days [1].",
    *,
    done_reason: str = "stop",
    prompt: int = 900,
    answer: int = 6,
) -> dict[str, object]:
    return {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt,
        "eval_count": answer,
    }


class StubServer:
    """Answers /api/version and /api/chat the way it is told to, and records every request."""

    def __init__(
        self, chat: httpx.Response | Exception | None = None, *, version: str | None = "0.34.2"
    ) -> None:
        self.chat = chat if chat is not None else httpx.Response(200, json=reply())
        self.version = version
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/version":
            if self.version is None:
                return httpx.Response(404, text="404 page not found")
            return httpx.Response(200, json={"version": self.version})
        if isinstance(self.chat, Exception):
            raise self.chat
        return self.chat

    def chat_bodies(self) -> list[object]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/api/chat"]


def chat_model(server: StubServer) -> OllamaChat:
    return OllamaChat(
        host="http://127.0.0.1:11434",
        model="llama3.2:3b",
        context_tokens=4096,
        max_answer_tokens=512,
        temperature=0.0,
        timeout_s=30,
        keep_alive="10m",
        transport=httpx.MockTransport(server),
    )


def complete(model: OllamaChat) -> Completion:
    return model.complete(system="Answer from the sources.", user="How long?")


def test_tells_ollama_never_to_truncate_or_shift_and_sets_the_limits() -> None:
    server = StubServer()
    completion = complete(chat_model(server))
    assert completion == Completion(text="Five days [1].", prompt_tokens=900, answer_tokens=6)
    assert server.chat_bodies() == [
        {
            "model": "llama3.2:3b",
            "messages": [
                {"role": "system", "content": "Answer from the sources."},
                {"role": "user", "content": "How long?"},
            ],
            "stream": False,
            "truncate": False,
            "shift": False,
            "keep_alive": "10m",
            "options": {"num_ctx": 4096, "num_predict": 512, "temperature": 0.0},
        }
    ]


def test_checks_the_server_version_once() -> None:
    server = StubServer()
    model = chat_model(server)
    complete(model)
    complete(model)
    assert [request.url.path for request in server.requests] == [
        "/api/version",
        "/api/chat",
        "/api/chat",
    ]


@pytest.mark.parametrize("version", ["0.12.5", "0.9.0", "0.0.0"])
def test_a_server_that_would_ignore_the_settings_is_refused(version: str) -> None:
    server = StubServer(version=version)
    with pytest.raises(ModelUnavailableError, match=f"Ollama {version} is too old") as caught:
        complete(chat_model(server))
    assert caught.value.hint is not None
    assert "update Ollama" in caught.value.hint
    assert server.chat_bodies() == []


@pytest.mark.parametrize("version", ["0.12.6", "0.34.2", "1.0.0-rc1"])
def test_servers_from_0_12_6_on_are_accepted(version: str) -> None:
    assert complete(chat_model(StubServer(version=version))).text == "Five days [1]."


@pytest.mark.parametrize(("version", "problem"), [("dev", "it says 'dev'"), (None, "(404)")])
def test_a_version_that_cannot_be_read_is_refused(version: str | None, problem: str) -> None:
    with pytest.raises(ModelUnavailableError, match=re.escape(problem)):
        complete(chat_model(StubServer(version=version)))


@pytest.mark.parametrize("body", [TOO_LONG_0_34, TOO_LONG_0_12], ids=["0.34", "0.12.6"])
def test_a_prompt_that_does_not_fit_raises_instead_of_being_cut(body: str) -> None:
    server = StubServer(httpx.Response(400, text=body))
    with pytest.raises(PromptTooLongError, match=r"4,096-token context of llama3\.2:3b") as caught:
        complete(chat_model(server))
    assert caught.value.hint is not None
    assert "top_k" in caught.value.hint


def test_the_prompt_size_is_given_when_ollama_reports_it() -> None:
    server = StubServer(httpx.Response(400, text=TOO_LONG_0_34))
    with pytest.raises(PromptTooLongError, match="the prompt of 6,046 tokens does not fit"):
        complete(chat_model(server))


def test_an_answer_stopped_at_the_answer_limit_raises() -> None:
    server = StubServer(httpx.Response(200, json=reply(done_reason="length", answer=512)))
    with pytest.raises(AnswerTooLongError, match="reached the limit of 512 tokens") as caught:
        complete(chat_model(server))
    assert caught.value.hint == "raise [ollama] chat_max_answer_tokens"


def test_an_answer_stopped_at_the_end_of_the_context_raises() -> None:
    server = StubServer(
        httpx.Response(200, json=reply(done_reason="length", prompt=3939, answer=157))
    )
    with pytest.raises(AnswerTooLongError, match="after a 3,939-token prompt"):
        complete(chat_model(server))


def test_a_reply_that_needed_more_than_the_context_raises() -> None:
    server = StubServer(httpx.Response(200, json=reply(prompt=3939, answer=400)))
    with pytest.raises(GenerationError, match="part of the prompt was dropped"):
        complete(chat_model(server))


@pytest.mark.parametrize(
    ("response", "problem"),
    [
        (httpx.Response(200, json=reply(content="  ")), "empty reply"),
        (httpx.Response(200, json=reply(done_reason="unload")), "stopped before finishing"),
        (httpx.Response(200, json={"message": {"content": "Hi"}}), "no answer or token counts"),
        (httpx.Response(200, text="<html>"), "not JSON: <html>"),
        (httpx.Response(200, json=["a list"]), "unexpected reply"),
        (httpx.Response(500, json={"error": "out of memory"}), "out of memory (500)"),
        (httpx.Response(502, text=""), "Bad Gateway (502)"),
    ],
)
def test_unusable_replies_raise_generation_error(response: httpx.Response, problem: str) -> None:
    with pytest.raises(GenerationError, match=re.escape(problem)):
        complete(chat_model(StubServer(response)))


def test_a_model_that_is_not_pulled_is_unavailable() -> None:
    server = StubServer(httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"}))
    with pytest.raises(ModelUnavailableError) as caught:
        complete(chat_model(server))
    assert caught.value.hint == "run `ollama pull llama3.2:3b`"


@pytest.mark.parametrize(
    ("error", "problem"),
    [
        (httpx.ConnectError("connection refused"), "cannot reach Ollama"),
        (httpx.ReadTimeout("timed out"), "did not answer within 30 s"),
        (httpx.RemoteProtocolError("server disconnected"), "lost the connection"),
    ],
)
def test_connection_problems_mean_the_model_is_unavailable(error: Exception, problem: str) -> None:
    with pytest.raises(ModelUnavailableError, match=problem):
        complete(chat_model(StubServer(error)))
