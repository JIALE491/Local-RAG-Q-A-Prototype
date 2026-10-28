"""Chat model backed by a local Ollama model, called through Ollama's REST API.

By default Ollama loses text without saying so. A prompt longer than the context window
is cut from the front, and an answer that reaches the end of the window makes room by
dropping part of the prompt ("context shift"); either way the reply looks normal. This
adapter turns both off: a prompt that does not fit raises PromptTooLongError, and an answer
that runs out of room raises AnswerTooLongError instead of coming back cut off.

The two request fields that do this, `truncate` and `shift`, exist for chat since Ollama
0.12.6. Older servers silently ignore them, so the server's version is checked before the
first request. The official Python client cannot send them for chat, which is why this
adapter calls /api/chat with httpx.
"""

import re

import httpx

from rag.domain.errors import (
    AnswerTooLongError,
    GenerationError,
    ModelUnavailableError,
    PromptTooLongError,
)
from rag.domain.models import Completion

MINIMUM_VERSION = (0, 12, 6)
"""The first Ollama release whose chat API accepts `truncate` and `shift`."""

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_TOO_LONG = re.compile(r"exceed.*context", re.IGNORECASE | re.DOTALL)
_TOKENS = re.compile(r"\((\d+) tokens\)")


class OllamaChat:
    """Answers with an Ollama chat model. Call close() when done."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        context_tokens: int,
        max_answer_tokens: int,
        temperature: float,
        timeout_s: float,
        keep_alive: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._host = host
        self._model = model
        self._context_tokens = context_tokens
        self._max_answer_tokens = max_answer_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._keep_alive = keep_alive
        self._client = httpx.Client(base_url=host, timeout=timeout_s, transport=transport)
        self._version_checked = False

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, *, system: str, user: str) -> Completion:
        if not self._version_checked:
            self._check_version()
            self._version_checked = True
        response = self._send(
            "POST",
            "/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "truncate": False,  # a prompt that does not fit is an error, not cut
                "shift": False,  # an answer that fills the context stops, keeping the prompt
                "keep_alive": self._keep_alive,
                "options": {
                    "num_ctx": self._context_tokens,
                    "num_predict": self._max_answer_tokens,
                    "temperature": self._temperature,
                },
            },
        )
        if response.is_error:
            raise self._chat_error(response)
        return self._completion(_json(response))

    def close(self) -> None:
        self._client.close()

    def _check_version(self) -> None:
        response = self._send("GET", "/api/version")
        if response.is_error:
            raise ModelUnavailableError(
                f"cannot tell which version of Ollama runs at {self._host}: "
                f"{_error_text(response)} ({response.status_code})"
            )
        version = str(_json(response).get("version", ""))
        match = _VERSION.match(version)
        if match is None:
            raise ModelUnavailableError(
                f"cannot tell which version of Ollama runs at {self._host} (it says {version!r})"
            )
        if tuple(int(part) for part in match.groups()) < MINIMUM_VERSION:
            raise ModelUnavailableError(
                f"Ollama {version} is too old: answering needs 0.12.6 or newer",
                hint="update Ollama; older versions ignore the setting that stops them from "
                "silently cutting a prompt that is too long",
            )

    def _send(self, method: str, path: str, *, json: object = None) -> httpx.Response:
        try:
            return self._client.request(method, path, json=json)
        except httpx.ConnectError as exc:
            raise ModelUnavailableError(
                f"cannot reach Ollama at {self._host}",
                hint="start the Ollama app, or check [ollama] host in config.toml",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ModelUnavailableError(
                f"Ollama did not answer within {self._timeout_s:g} s",
                hint="the first request also loads the model; raise [ollama] timeout_s "
                "if this keeps happening",
            ) from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError(f"lost the connection to Ollama: {exc}") from exc

    def _chat_error(self, response: httpx.Response) -> Exception:
        if response.status_code == 404:
            return ModelUnavailableError(
                f"the model {self._model!r} is not available in Ollama",
                hint=f"run `ollama pull {self._model}`",
            )
        if _TOO_LONG.search(response.text):
            counts = _TOKENS.findall(response.text)
            size = f" of {int(counts[0]):,} tokens" if counts else ""
            return PromptTooLongError(
                f"the prompt{size} does not fit in the {self._context_tokens:,}-token context "
                f"of {self._model}",
                hint="give the model fewer chunks ([retrieval] top_k, or -k), or raise "
                "[ollama] chat_context_tokens",
            )
        return GenerationError(
            f"Ollama could not answer: {_error_text(response)} ({response.status_code})"
        )

    def _completion(self, reply: dict[str, object]) -> Completion:
        message = reply.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        prompt_tokens = reply.get("prompt_eval_count")
        answer_tokens = reply.get("eval_count")
        if not (
            isinstance(text, str)
            and isinstance(prompt_tokens, int)
            and isinstance(answer_tokens, int)
        ):
            raise GenerationError(f"Ollama's reply has no answer or token counts: {reply!r:.300}")
        done_reason = reply.get("done_reason")
        if done_reason == "length":
            raise self._cut_off(prompt_tokens, answer_tokens)
        if done_reason != "stop":
            raise GenerationError(
                f"the model stopped before finishing its answer ({done_reason!r})"
            )
        if prompt_tokens + answer_tokens > self._context_tokens:
            raise GenerationError(
                f"the prompt and answer took {prompt_tokens + answer_tokens:,} tokens, more than "
                f"the {self._context_tokens:,}-token context, so part of the prompt was dropped"
            )
        if not text.strip():
            raise GenerationError("the model returned an empty reply")
        return Completion(text=text, prompt_tokens=prompt_tokens, answer_tokens=answer_tokens)

    def _cut_off(self, prompt_tokens: int, answer_tokens: int) -> AnswerTooLongError:
        if prompt_tokens + answer_tokens >= self._context_tokens:
            return AnswerTooLongError(
                f"the answer filled the {self._context_tokens:,}-token context after a "
                f"{prompt_tokens:,}-token prompt and was cut off after {answer_tokens:,} tokens",
                hint="give the model fewer chunks ([retrieval] top_k, or -k), or raise "
                "[ollama] chat_context_tokens",
            )
        return AnswerTooLongError(
            f"the answer reached the limit of {self._max_answer_tokens:,} tokens and was cut off",
            hint="raise [ollama] chat_max_answer_tokens",
        )


def _json(response: httpx.Response) -> dict[str, object]:
    try:
        data = response.json()
    except ValueError as exc:
        raise GenerationError(
            f"Ollama sent a reply that is not JSON: {response.text:.200}"
        ) from exc
    if not isinstance(data, dict):
        raise GenerationError(f"Ollama sent an unexpected reply: {data!r:.200}")
    return data


def _error_text(response: httpx.Response) -> str:
    """Ollama's error message, or the raw body if it is not Ollama's JSON error."""
    try:
        data = response.json()
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    return str(error) if error else (response.text.strip()[:300] or response.reason_phrase)
