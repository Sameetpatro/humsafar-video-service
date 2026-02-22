# app/services/__init__.py
# Exports call_openrouter for use in main.py

import logging
import os
import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


async def call_openrouter(messages: list) -> str:
    """
    Send a list of OpenAI-format messages to OpenRouter.
    Returns the assistant reply text.
    Raises RuntimeError on any failure.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    logger.info(f"[OpenRouter] Calling model={OPENROUTER_MODEL} with {len(messages)} messages")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
            },
        )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    reply = resp.json()["choices"][0]["message"]["content"].strip()
    logger.info(f"[OpenRouter] Reply ({len(reply)} chars): '{reply[:80]}'")
    return reply