# app/routers/voice.py
# Exposes:
#   POST /transcribe  — user speaks a question → returns transcribed text
#                       The frontend then sends that text to POST /chat

import logging
import os

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
STT_URL        = "https://api.sarvam.ai/speech-to-text"


class TranscribeResponse(BaseModel):
    text: str
    language_code: str = ""


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language_code: str = Form("en-IN"),
):
    """
    Accept an audio recording from the user (wav/mp3/webm),
    return transcribed text via Sarvam STT.
    The frontend passes this text to POST /chat to get a reply,
    and optionally to POST /generate to get a video summary.
    """
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SARVAM_API_KEY is not configured",
        )

    audio_bytes = await file.read()
    logger.info(
        f"[STT] Transcribing {len(audio_bytes):,} bytes "
        f"({file.content_type}) lang={language_code}"
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            STT_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={
                "file": (file.filename or "audio.wav", audio_bytes, file.content_type)
            },
            data={"language_code": language_code},
        )

    if resp.status_code != 200:
        logger.error(f"[STT] Error {resp.status_code}: {resp.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Sarvam STT error {resp.status_code}: {resp.text}",
        )

    data          = resp.json()
    text          = data.get("transcript", "").strip()
    detected_lang = data.get("language_code", language_code)
    logger.info(f"[STT] → '{text[:80]}'")
    return TranscribeResponse(text=text, language_code=detected_lang)