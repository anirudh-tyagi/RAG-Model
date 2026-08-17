"""Ollama Cloud client for text and vision calls.

Differences from the old code, which built a ``ChatOllama`` at import time and
called ``exit()`` if the key was missing — taking the whole server down with it:

* a missing or bad key raises :class:`LlmError` on the request that needs it,
  so the API still boots, still serves ``/health``, and still reports why;
* streaming is a first-class path, so answers appear token by token;
* transient network failures are retried with backoff.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

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


class LlmError(RuntimeError):
    """Any failure talking to the model provider, safe to surface to the client."""


class LlmClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    # --- wiring ----------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.ollama_api_key:
            raise LlmError(
                "OLLAMA_API_KEY is not set. Add it to .env — get a key at "
                "https://ollama.com/settings/keys"
            )
        from ollama import AsyncClient

        self._client = AsyncClient(
            host=self._settings.ollama_base_url,
            headers={"Authorization": f"Bearer {self._settings.ollama_api_key.get_secret_value()}"},
            timeout=self._settings.llm_timeout_s,
        )
        return self._client

    @property
    def configured(self) -> bool:
        return self._settings.ollama_api_key is not None

    async def health(self) -> bool:
        """Cheap liveness probe: a one-token completion."""
        try:
            await self.complete([{"role": "user", "content": "ping"}], max_tokens=1)
            return True
        except Exception as exc:
            log.warning("llm_unreachable", error=str(exc))
            return False

    def _options(self, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else self._settings.llm_temperature
            )
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return options

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=6),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            reraise=True,
        )

    # --- text ------------------------------------------------------------

    async def complete(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        client = self._ensure_client()
        model = model or self._settings.llm_model
        try:
            async for attempt in self._retrying():
                with attempt:
                    response = await client.chat(
                        model=model,
                        messages=list(messages),
                        stream=False,
                        options=self._options(temperature, max_tokens),
                    )
                    return str(response.message.content or "")
        except Exception as exc:
            raise LlmError(f"{model} request failed: {exc}") from exc
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
        client = self._ensure_client()
        model = model or self._settings.llm_model
        try:
            parts = await client.chat(
                model=model,
                messages=list(messages),
                stream=True,
                options=self._options(temperature, None),
            )
            async for part in parts:
                text = part.message.content if part.message else None
                if text:
                    yield text
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"{model} stream failed: {exc}") from exc

    # --- vision ----------------------------------------------------------

    async def describe_image(self, image_path: Path, prompt: str) -> str:
        """Caption one figure with the vision model."""
        client = self._ensure_client()
        model = self._settings.vision_model
        try:
            async for attempt in self._retrying():
                with attempt:
                    response = await client.chat(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                                "images": [str(image_path)],
                            }
                        ],
                        stream=False,
                        options=self._options(0.1, None),
                    )
                    return str(response.message.content or "").strip()
        except Exception as exc:
            raise LlmError(f"{model} vision request failed: {exc}") from exc
        raise LlmError("retry loop exhausted without a response")  # pragma: no cover


_llm: LlmClient | None = None


def get_llm() -> LlmClient:
    global _llm
    if _llm is None:
        _llm = LlmClient(get_settings())
    return _llm
