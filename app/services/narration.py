# app/services/narration.py
# Generates narration text via OpenRouter, then synthesises WAV via Sarvam TTS.
# Reuses the same LLM/TTS approach as the main backend.

import base64
import logging
import os

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY", "")

TTS_URL   = "https://api.sarvam.ai/text-to-speech"
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL",   "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "ritu")
MAX_CHARS = 500


async def generate_narration_text(prompt: str, site_name: str) -> str:
    """
    Ask the LLM to produce a short video narration script (≤ 4 sentences).
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    system = (
        "You are HUMSAFAR, a cinematic heritage guide narrator. "
        "Write a short, engaging video narration (2–4 sentences, no markdown, "
        "no bullet points) about the heritage site provided. "
        "Focus on visual imagery, history, and atmosphere. "
        "Do not hallucinate facts."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Site: {site_name}\nContext: {prompt}"},
    ]

    logger.info(f"[Narration] LLM call — site={site_name}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "openai/gpt-4o-mini", "messages": messages},
        )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    text = resp.json()["choices"][0]["message"]["content"].strip()
    logger.info(f"[Narration] Text ({len(text)} chars): '{text[:80]}'")
    return text


async def synthesise_wav(text: str, language_code: str = "en-IN") -> bytes:
    """
    Send narration text to Sarvam TTS, return raw WAV bytes.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "…"
        logger.warning(f"[TTS] Truncated to {len(text)} chars")

    logger.info(f"[TTS] Synthesising {len(text)} chars, lang={language_code}")

    payload = {
        "inputs":               [text],
        "target_language_code": language_code,
        "speaker":              TTS_SPEAKER,
        "model":                TTS_MODEL,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            TTS_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Sarvam TTS error {resp.status_code}: {resp.text}")

    audios = resp.json().get("audios", [])
    if not audios:
        raise RuntimeError("Sarvam TTS returned empty audio list")

    wav = base64.b64decode(audios[0])
    logger.info(f"[TTS] Synthesised {len(wav):,} bytes")
    return wav