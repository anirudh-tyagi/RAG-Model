"""Voice input: audio upload → English text."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from rag.api.deps import SettingsDep, TranscriberDep
from rag.logging import get_logger
from rag.schemas import TranscriptionOut

router = APIRouter(tags=["audio"])
log = get_logger(__name__)

MAX_AUDIO_MB = 25
#: Starlette renamed the 413 constant; the integer works across versions.
HTTP_413 = 413

AudioUpload = Annotated[UploadFile, File(description="A recording to transcribe")]


@router.post("/transcribe", response_model=TranscriptionOut)
async def transcribe(
    audio: AudioUpload,
    transcriber: TranscriberDep,
    settings: SettingsDep,
) -> TranscriptionOut:
    """Transcribe a recording, translating to English when it isn't already.

    Accepts whatever the browser's MediaRecorder produces (webm/opus by
    default) — faster-whisper decodes it directly, so there is no ffmpeg
    conversion step and no `pydub` in the dependency tree.
    """
    data = await audio.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded audio is empty.")
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(HTTP_413, f"Audio exceeds the {MAX_AUDIO_MB}MB limit.")

    try:
        result = await transcriber.transcribe(data)
    except Exception as exc:
        log.exception("transcription_failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Could not transcribe audio: {exc}",
        ) from exc

    if not result.text:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "No speech detected. Try recording again, closer to the microphone.",
        )
    log.info(
        "transcribed",
        language=result.language,
        duration_s=result.duration_s,
        model=settings.whisper_model,
    )
    return result
