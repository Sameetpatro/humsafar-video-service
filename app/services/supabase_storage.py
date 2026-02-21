# app/services/supabase_storage.py
# All Supabase Storage I/O for the video microservice.
# supabase-py is sync — wrapped in asyncio.to_thread throughout.

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET      = os.getenv("SUPABASE_BUCKET", "videos")
STORAGE_PREFIX       = "generated"


def _client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    from supabase import create_client  # type: ignore
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def _object_path(job_id: str) -> str:
    return f"{STORAGE_PREFIX}/{job_id}.mp4"


def _public_url(job_id: str) -> str:
    return (
        f"{SUPABASE_URL.rstrip('/')}"
        f"/storage/v1/object/public/{SUPABASE_BUCKET}"
        f"/{_object_path(job_id)}"
    )


async def upload_video(local_path: str, job_id: str) -> str:
    """
    Upload local MP4 → Supabase Storage.
    Returns public URL.
    Raises RuntimeError on any failure.
    """
    p = Path(local_path)
    if not p.exists():
        raise RuntimeError(f"File not found before upload: {local_path}")

    size = p.stat().st_size
    logger.info(
        f"[Storage] Uploading {size:,} B → "
        f"bucket={SUPABASE_BUCKET} path={_object_path(job_id)}"
    )

    def _sync():
        c = _client()
        with open(local_path, "rb") as f:
            data = f.read()
        c.storage.from_(SUPABASE_BUCKET).upload(
            path=_object_path(job_id),
            file=data,
            file_options={"content-type": "video/mp4", "upsert": "true"},
        )

    try:
        await asyncio.to_thread(_sync)
    except Exception as exc:
        raise RuntimeError(f"Supabase upload failed (job={job_id}): {exc}") from exc

    url = _public_url(job_id)
    logger.info(f"[Storage] Upload complete ✓ → {url}")
    return url


async def delete_video(job_id: str) -> bool:
    """Delete from Supabase. Returns True if object was found and removed."""
    obj = _object_path(job_id)
    logger.info(f"[Storage] Deleting {obj}")

    def _sync():
        return _client().storage.from_(SUPABASE_BUCKET).remove([obj])

    try:
        result = await asyncio.to_thread(_sync)
    except Exception as exc:
        raise RuntimeError(f"Supabase delete failed (job={job_id}): {exc}") from exc

    if not result:
        logger.warning(f"[Storage] Object not found: {obj}")
        return False

    logger.info(f"[Storage] Deleted ✓ {obj}")
    return True