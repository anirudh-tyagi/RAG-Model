"""Speech to text with faster-whisper.

Replaces three moving parts from the old code:

* ``speech_recognition``'s ``recognize_google``, which called an undocumented,
  unauthenticated Google endpoint with no rate limit guarantees;
* ``googletrans``, an unofficial scrape of Google Translate — Whisper's
  ``translate`` task turns any supported language straight into English, so the
  separate transcribe-then-translate hop in ``STT.py`` is gone;
* ``pydub``, and with it the system ``ffmpeg`` requirement — faster-whisper
  decodes the browser's webm/opus stream directly via PyAV.

CTranslate2 also means no torch: the ``small`` model is roughly 250MB and runs
comfortably on CPU.
"""

from __future__ import annotations

import asyncio
import io
import threading
from typing import TYPE_CHECKING

from rag.config import Settings, get_settings
from rag.logging import get_logger
from rag.schemas import TranscriptionOut

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = get_logger(__name__)


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                log.info(
                    "loading_whisper",
                    model=self._settings.whisper_model,
                    compute_type=self._settings.whisper_compute_type,
                )
                self._model = WhisperModel(
                    self._settings.whisper_model,
                    device="cpu",
                    compute_type=self._settings.whisper_compute_type,
                )

    def _transcribe(self, audio: bytes) -> TranscriptionOut:
        self._load()
        assert self._model is not None
        segments, info = self._model.transcribe(
            io.BytesIO(audio),
            task=self._settings.whisper_task,
            beam_size=5,
            # Skips silence, which is most of a short browser recording.
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionOut(
            text=text,
            language=getattr(info, "language", None),
            duration_s=getattr(info, "duration", None),
        )

    async def transcribe(self, audio: bytes) -> TranscriptionOut:
        if not audio:
            return TranscriptionOut(text="")
        return await asyncio.to_thread(self._transcribe, audio)


_transcriber: Transcriber | None = None


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber(get_settings())
    return _transcriber
