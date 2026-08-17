"""LLM client, speaking the OpenAI chat-completions dialect.

That dialect is the lingua franca now — Groq, Ollama Cloud, OpenAI, Together,
vLLM and most self-hosted servers all expose it — so the provider is a base-URL
and model-name change rather than a rewrite. Defaults point at Groq.

Differences from the original code, which built a ``ChatOllama`` at import time
and called ``exit()`` when the key was missing, taking the server down with it:

* a missing or bad key raises :class:`LlmError` on the request that needs it,
  so the API still boots, still serves ``/health``, and still reports why;
* streaming is a first-class path, so answers appear token by token;
* transient network failures are retried with backoff.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    RateLimitError,
)

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessageParam,
    )
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.config import Settings, get_settings
from rag.logging import get_logger

log = get_logger(__name__)

Messages = Sequence[dict[str, Any]]


def _as_params(messages: Messages) -> list[ChatCompletionMessageParam]:
    """Hand our plain dicts to the SDK's TypedDict-shaped parameter type.

    The dicts are built in ``prompts.py`` and are already the right shape; this
    is a typing bridge, not a conversion.
    """
    return cast("list[ChatCompletionMessageParam]", list(messages))


#: Failures worth another attempt: the network blipped or the provider is busy.
RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)


class LlmError(RuntimeError):
    """Any failure talking to the model provider, safe to surface to the client."""


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat: AsyncOpenAI | None = None
        self._vision: AsyncOpenAI | None = None

    # --- wiring ----------------------------------------------------------

    def _client(self) -> AsyncOpenAI:
        if self._chat is None:
            if not self._settings.llm_api_key:
                raise LlmError(
                    "LLM_API_KEY is not set. Add it to .env — for Groq, get one at "
                    "https://console.groq.com/keys"
                )
            self._chat = AsyncOpenAI(
                api_key=self._settings.llm_api_key.get_secret_value(),
                base_url=self._settings.llm_base_url,
                timeout=self._settings.llm_timeout_s,
                max_retries=0,  # tenacity owns retries, so they aren't doubled
            )
        return self._chat

    def _vision_client(self) -> AsyncOpenAI:
        if self._vision is None:
            key = self._settings.effective_vision_api_key
            if not key:
                raise LlmError("No API key configured for the vision model.")
            self._vision = AsyncOpenAI(
                api_key=key.get_secret_value(),
                base_url=self._settings.effective_vision_base_url,
                timeout=self._settings.llm_timeout_s,
                max_retries=0,
            )
        return self._vision

    @property
    def configured(self) -> bool:
        return self._settings.llm_api_key is not None

    async def health(self) -> bool:
        """Cheap liveness probe: list the provider's models."""
        try:
            await self._client().models.list()
            return True
        except Exception as exc:
            log.warning("llm_unreachable", error=str(exc))
            return False

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=8),
            retry=retry_if_exception_type(RETRYABLE),
            reraise=True,
        )

    @staticmethod
    def _describe(exc: Exception) -> str:
        """Turn a provider error into something worth showing a user."""
        if isinstance(exc, APIStatusError):
            if exc.status_code == 401:
                return "the API key was rejected (401)"
            if exc.status_code == 404:
                return "the model was not found on this provider (404)"
            if exc.status_code == 429:
                return "rate limited by the provider (429)"
            return f"provider returned {exc.status_code}"
        if isinstance(exc, APITimeoutError):
            return "the request timed out"
        if isinstance(exc, APIConnectionError):
            return "could not reach the provider"
        return str(exc)

    # --- text ------------------------------------------------------------

    async def complete(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model = model or self._settings.llm_model
        client = self._client()
        try:
            async for attempt in self._retrying():
                with attempt:
                    response: ChatCompletion = await client.chat.completions.create(
                        model=model,
                        messages=_as_params(messages),
                        temperature=(
                            temperature
                            if temperature is not None
                            else self._settings.llm_temperature
                        ),
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    return response.choices[0].message.content or ""
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"{model}: {self._describe(exc)}") from exc
        raise LlmError("retry loop exhausted without a response")  # pragma: no cover

    async def stream(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer text as it is generated.

        Retries deliberately do not wrap this: once tokens have reached the
        client, replaying the call would duplicate output.
        """
        model = model or self._settings.llm_model
        client = self._client()
        try:
            stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
                model=model,
                messages=_as_params(messages),
                temperature=(
                    temperature if temperature is not None else self._settings.llm_temperature
                ),
                stream=True,
            )
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        yield text
            finally:
                # The consumer may stop early — a user pressing stop, or a
                # disconnect. Without this the underlying HTTP response is left
                # for the garbage collector to tear down, which surfaces as
                # "generator didn't stop after athrow()" and leaks the connection.
                await stream.close()
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"{model}: {self._describe(exc)}") from exc

    # --- vision ----------------------------------------------------------

    async def describe_image(self, image_path: Path, prompt: str) -> str:
        """Caption one figure with the vision model.

        The image is inlined as a base64 data URL, which every OpenAI-compatible
        multimodal endpoint accepts and which avoids needing to host the file
        anywhere the provider can reach.
        """
        model = self._settings.vision_model
        client = self._vision_client()
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        try:
            # Reading off the loop: figures can be a few MB each.
            raw = await asyncio.to_thread(image_path.read_bytes)
            encoded = base64.b64encode(raw).decode("ascii")
        except OSError as exc:
            raise LlmError(f"could not read {image_path.name}: {exc}") from exc

        try:
            async for attempt in self._retrying():
                with attempt:
                    response: ChatCompletion = await client.chat.completions.create(
                        model=model,
                        messages=_as_params(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                                        },
                                    ],
                                }
                            ]
                        ),
                        temperature=0.1,
                        stream=False,
                    )
                    return (response.choices[0].message.content or "").strip()
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"{model} (vision): {self._describe(exc)}") from exc
        raise LlmError("retry loop exhausted without a response")  # pragma: no cover


_llm: LlmClient | None = None


def get_llm() -> LlmClient:
    global _llm
    if _llm is None:
        _llm = LlmClient(get_settings())
    return _llm
