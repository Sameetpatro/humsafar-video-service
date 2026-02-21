# app/routers/video.py
# Exposes:
#   POST /generate    — enqueue a video job
#   GET  /status/{id} — poll job progress

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from app.job_manager                    import jobs
from app.services.narration             import generate_narration_text, synthesise_wav
from app.services.ffmpeg_pipeline       import run_ffmpeg
from app.services.supabase_storage      import upload_video

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video"])

# One worker at a time — prevents memory spikes on free tier
_ffmpeg_pool = ThreadPoolExecutor(max_workers=1)


# ── Models ────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:    str
    bot_text:  str          # narration text provided by main backend
    site_id:   str
    site_name: str
    language_code: str = "en-IN"


class GenerateResponse(BaseModel):
    job_id: str
    status: str = "generating"


class StatusResponse(BaseModel):
    job_id:    str
    status:    str
    progress:  int         = 0
    video_url: str | None  = None
    message:   str         = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Enqueue a video generation job.
    Returns job_id immediately; client polls /status/{job_id}.
    """
    job = jobs.create()
    logger.info(
        f"[/generate] Enqueued job_id={job.job_id} "
        f"site={req.site_name} lang={req.language_code}"
    )
    background_tasks.add_task(
        _run_pipeline,
        job_id        = job.job_id,
        prompt        = req.prompt,
        bot_text      = req.bot_text,
        site_id       = req.site_id,
        site_name     = req.site_name,
        language_code = req.language_code,
    )
    return GenerateResponse(job_id=job.job_id)


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return StatusResponse(
        job_id    = job.job_id,
        status    = job.status,
        progress  = job.progress,
        video_url = job.video_url,
        message   = job.message,
    )


# ── Background pipeline ───────────────────────────────────────────────────────

async def _run_pipeline(
    job_id: str,
    prompt: str,
    bot_text: str,
    site_id: str,
    site_name: str,
    language_code: str,
):
    """
    Full pipeline:
      1. Generate narration WAV (TTS)
      2. Run FFmpeg → temp MP4
      3. Upload to Supabase
      4. Delete local temp file
    """
    def _upd(progress: int, message: str = ""):
        jobs.update(job_id, progress=progress, message=message)

    # Temp file for the MP4 output — lives only until upload succeeds
    tmp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_mp4.close()
    output_path = tmp_mp4.name

    try:
        # ── Stage 1: TTS ─────────────────────────────────────────────────
        _upd(10, "Generating narration…")

        # Prefer bot_text if provided (main backend already generated it),
        # otherwise fall back to LLM generation here.
        if bot_text and bot_text.strip():
            narration_text = bot_text.strip()
            logger.info(f"[Pipeline/{job_id}] Using provided bot_text for narration")
        else:
            narration_text = await generate_narration_text(prompt, site_name)

        wav_bytes = await synthesise_wav(narration_text, language_code)
        _upd(35, "Narration ready")

        # ── Stage 2: FFmpeg (runs in thread pool, never blocks event loop) ─
        _upd(40, "Rendering video…")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _ffmpeg_pool,
            run_ffmpeg,
            wav_bytes,
            site_id,
            output_path,
        )
        _upd(75, "Video rendered")

        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg succeeded but output file is missing")

        # ── Stage 3: Upload ───────────────────────────────────────────────
        _upd(80, "Uploading to Supabase…")
        public_url = await upload_video(output_path, job_id)
        _upd(98, "Upload complete")

        # ── Stage 4: Delete local file ────────────────────────────────────
        try:
            os.unlink(output_path)
            logger.info(f"[Pipeline/{job_id}] Temp file deleted")
        except OSError as e:
            logger.warning(f"[Pipeline/{job_id}] Could not delete temp file: {e}")

        # ── Done ──────────────────────────────────────────────────────────
        jobs.update(job_id, status="ready", progress=100, video_url=public_url)
        logger.info(f"[Pipeline/{job_id}] ✓ Complete → {public_url}")

    except Exception as exc:
        logger.error(f"[Pipeline/{job_id}] FAILED: {exc}", exc_info=True)
        # Best-effort cleanup
        try:
            os.unlink(output_path)
        except OSError:
            pass
        jobs.update(job_id, status="failed", progress=0, message=str(exc))