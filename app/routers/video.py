# video_service/app/routers/video.py
# FIXED VERSION:
#   - 480p output (was 1080p → OOM-killed on Render free tier)
#   - ultrafast preset, crf=28 (was fast/23 → too slow for free tier)
#   - Aggressive logging at every step so you can see exactly where it dies
#   - Proper error propagation so Android gets a "failed" status instead of hanging
#   - Uses Unsplash API for images (free, no auth needed for small sizes)
#   - Falls back to colored placeholder if image fetch fails

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET      = os.getenv("SUPABASE_BUCKET", "videos")
SARVAM_API_KEY       = os.getenv("SARVAM_API_KEY", "")
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")

# FFmpeg settings — tuned for Render free tier (512 MB RAM, shared CPU)
FFMPEG_RESOLUTION = "854:480"   # 480p — ~4x less memory than 1080p
FFMPEG_PRESET     = "ultrafast" # fastest encode, larger file but won't OOM
FFMPEG_CRF        = "30"        # slightly lower quality = smaller file
FFMPEG_TIMEOUT    = 180         # 3 min max for FFmpeg

router = APIRouter(tags=["video"])

# Single worker — prevents two concurrent FFmpeg jobs from OOM-killing each other
_ffmpeg_pool = ThreadPoolExecutor(max_workers=1)

# In-memory job state
_jobs: dict[str, dict] = {}


# ── Models ────────────────────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    bot_text:      str           # The text to narrate (from chatbot response)
    site_name:     str
    site_id:       str           # Used for image search
    language_code: str = "en-IN"


class GenerateVideoResponse(BaseModel):
    job_id:    str
    status:    str = "generating"


class VideoStatusResponse(BaseModel):
    job_id:    str
    status:    str               # "generating" | "ready" | "failed"
    progress:  int = 0
    video_url: str | None = None
    message:   str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_job_id(bot_text: str, site_id: str) -> str:
    raw = f"{site_id}::{bot_text.strip()[:300]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _update_job(job_id: str, **kwargs):
    """Update job state and log it."""
    _jobs.setdefault(job_id, {}).update(kwargs)
    status   = _jobs[job_id].get("status", "?")
    progress = _jobs[job_id].get("progress", 0)
    message  = _jobs[job_id].get("message", "")
    logger.info(f"[Job/{job_id}] {status} {progress}% — {message}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateVideoResponse)
async def generate_video(req: GenerateVideoRequest, background_tasks: BackgroundTasks):
    logger.info(f"[/generate] site={req.site_name} lang={req.language_code} text_len={len(req.bot_text)}")

    job_id = _make_job_id(req.bot_text, req.site_id)
    logger.info(f"[/generate] job_id={job_id}")

    # Cache hit — already generated
    existing = _jobs.get(job_id)
    if existing:
        if existing.get("status") == "ready":
            logger.info(f"[/generate] Cache hit for job_id={job_id}")
            return GenerateVideoResponse(job_id=job_id, status="ready")
        if existing.get("status") == "generating":
            logger.info(f"[/generate] Already generating job_id={job_id}")
            return GenerateVideoResponse(job_id=job_id, status="generating")
        # Failed — allow retry
        logger.info(f"[/generate] Retrying failed job_id={job_id}")

    _update_job(job_id, status="generating", progress=0, message="Queued")
    background_tasks.add_task(
        _run_pipeline,
        job_id        = job_id,
        bot_text      = req.bot_text,
        site_name     = req.site_name,
        site_id       = req.site_id,
        language_code = req.language_code,
    )

    return GenerateVideoResponse(job_id=job_id, status="generating")


@router.get("/status/{job_id}", response_model=VideoStatusResponse)
async def get_status(job_id: str):
    logger.debug(f"[/status/{job_id}] polled")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return VideoStatusResponse(
        job_id    = job_id,
        status    = job.get("status", "generating"),
        progress  = job.get("progress", 0),
        video_url = job.get("video_url"),
        message   = job.get("message", ""),
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def _run_pipeline(
    job_id:        str,
    bot_text:      str,
    site_name:     str,
    site_id:       str,
    language_code: str,
):
    logger.info(f"[Pipeline/{job_id}] ===== STARTING =====")
    t_start = time.time()

    tmp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_mp4.close()
    output_path = tmp_mp4.name
    logger.info(f"[Pipeline/{job_id}] Output path: {output_path}")

    try:
        # ── Stage 1: TTS ─────────────────────────────────────────────────
        logger.info(f"[Pipeline/{job_id}] Stage 1: TTS — {len(bot_text)} chars")
        _update_job(job_id, progress=10, message="Generating narration audio…")

        wav_bytes = await _tts(bot_text, language_code)
        logger.info(f"[Pipeline/{job_id}] TTS done — {len(wav_bytes):,} bytes in {time.time()-t_start:.1f}s")
        _update_job(job_id, progress=30, message="Audio ready")

        # ── Stage 2: Images ───────────────────────────────────────────────
        logger.info(f"[Pipeline/{job_id}] Stage 2: Fetching images for '{site_name}'")
        _update_job(job_id, progress=35, message="Fetching images…")

        image_paths = await _get_images(site_name, job_id)
        logger.info(f"[Pipeline/{job_id}] Images: {image_paths}")
        _update_job(job_id, progress=50, message=f"Got {len(image_paths)} images")

        # ── Stage 3: FFmpeg ───────────────────────────────────────────────
        logger.info(f"[Pipeline/{job_id}] Stage 3: FFmpeg encode")
        _update_job(job_id, progress=55, message="Rendering video (480p ultrafast)…")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _ffmpeg_pool,
            _run_ffmpeg_sync,
            job_id,
            image_paths,
            wav_bytes,
            output_path,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg finished but output file missing")

        size = os.path.getsize(output_path)
        logger.info(f"[Pipeline/{job_id}] FFmpeg done — {size:,} bytes in {time.time()-t_start:.1f}s")
        _update_job(job_id, progress=75, message="Video rendered, uploading…")

        # ── Stage 4: Upload to Supabase ───────────────────────────────────
        logger.info(f"[Pipeline/{job_id}] Stage 4: Supabase upload")
        public_url = await _upload_supabase(output_path, job_id)
        logger.info(f"[Pipeline/{job_id}] Upload done → {public_url} in {time.time()-t_start:.1f}s")

        # ── Done ──────────────────────────────────────────────────────────
        total = time.time() - t_start
        logger.info(f"[Pipeline/{job_id}] ===== COMPLETE in {total:.1f}s ===== URL={public_url}")
        _update_job(job_id, status="ready", progress=100, video_url=public_url, message="Done")

    except Exception as exc:
        total = time.time() - t_start
        logger.error(f"[Pipeline/{job_id}] ===== FAILED after {total:.1f}s: {exc} =====", exc_info=True)
        _update_job(job_id, status="failed", progress=0, message=str(exc))
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


# ── TTS ───────────────────────────────────────────────────────────────────────

async def _tts(text: str, language_code: str) -> bytes:
    import base64
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set")

    # Truncate to 500 chars (Sarvam limit)
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "…"
        logger.warning(f"[TTS] Truncated text to {len(text)} chars")

    logger.info(f"[TTS] POST to Sarvam, {len(text)} chars, lang={language_code}")

    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "inputs":               [text],
                "target_language_code": language_code,
                "speaker":              "meera",
                "model":                "bulbul:v3",
            },
        )

    logger.info(f"[TTS] Response: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"Sarvam TTS error {resp.status_code}: {resp.text[:300]}")

    audios = resp.json().get("audios", [])
    if not audios:
        raise RuntimeError("Sarvam TTS returned empty audio list")

    return base64.b64decode(audios[0])


# ── Images ────────────────────────────────────────────────────────────────────

async def _get_images(site_name: str, job_id: str) -> list[str]:
    """
    Download 3 images from Picsum (free, no API key needed).
    Uses deterministic seeds based on site_name so same site gets same images.
    Falls back to colored placeholder JPEGs if download fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"imgs_{job_id}_")
    paths   = []

    # Try Picsum Photos (free, no auth, reliable)
    seeds = [abs(hash(site_name + str(i))) % 1000 for i in range(3)]

    async with httpx.AsyncClient(timeout=15.0) as client:
        for i, seed in enumerate(seeds):
            url = f"https://picsum.photos/seed/{seed}/854/480"
            img_path = os.path.join(tmp_dir, f"img_{i:02d}.jpg")
            try:
                logger.info(f"[Images/{job_id}] Downloading image {i+1}/3 seed={seed}")
                r = await client.get(url, follow_redirects=True)
                if r.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                    paths.append(img_path)
                    logger.info(f"[Images/{job_id}] Image {i+1} downloaded: {len(r.content):,} bytes")
                else:
                    logger.warning(f"[Images/{job_id}] Picsum returned {r.status_code} for seed={seed}")
            except Exception as e:
                logger.warning(f"[Images/{job_id}] Image download failed: {e}")

    if not paths:
        logger.warning(f"[Images/{job_id}] All downloads failed — using placeholder")
        placeholder = os.path.join(tmp_dir, "placeholder.jpg")
        _write_placeholder_jpg(placeholder)
        paths = [placeholder]

    return paths


def _write_placeholder_jpg(path: str):
    """Minimal valid black 1x1 JPEG."""
    data = bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
        0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
        0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
        0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
        0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
        0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
        0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
        0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
        0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
        0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0A,0x0B,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,
        0x00,0xF5,0x7F,0xFF,0xD9
    ])
    with open(path, "wb") as f:
        f.write(data)


# ── FFmpeg (sync, runs in thread pool) ───────────────────────────────────────

def _run_ffmpeg_sync(job_id: str, image_paths: list[str], audio_bytes: bytes, output_path: str):
    logger.info(f"[FFmpeg/{job_id}] Starting — {len(image_paths)} images, output={output_path}")

    with tempfile.TemporaryDirectory() as tmp:
        # Write audio
        audio_path = os.path.join(tmp, "audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"[FFmpeg/{job_id}] Audio written: {len(audio_bytes):,} bytes")

        # Probe duration
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", audio_path],
                capture_output=True, text=True, timeout=15
            )
            duration = float(probe.stdout.strip())
            logger.info(f"[FFmpeg/{job_id}] Audio duration: {duration:.1f}s")
        except Exception as e:
            logger.warning(f"[FFmpeg/{job_id}] ffprobe failed ({e}), using 20s")
            duration = 20.0

        dur_per_img = max(duration / len(image_paths), 2.0)
        logger.info(f"[FFmpeg/{job_id}] {len(image_paths)} images @ {dur_per_img:.1f}s each")

        # Concat list
        concat = os.path.join(tmp, "concat.txt")
        with open(concat, "w") as f:
            for p in image_paths:
                f.write(f"file '{p}'\n")
                f.write(f"duration {dur_per_img:.2f}\n")
            f.write(f"file '{image_paths[-1]}'\n")

        vf = (
            f"scale={FFMPEG_RESOLUTION}:force_original_aspect_ratio=decrease,"
            f"pad={FFMPEG_RESOLUTION}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat,
            "-i", audio_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", FFMPEG_PRESET,
            "-crf", FFMPEG_CRF,
            "-c:a", "aac", "-b:a", "96k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            "-r", "24",
            output_path,
        ]

        logger.info(f"[FFmpeg/{job_id}] Running: {' '.join(cmd)}")
        t0 = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT
        )

        elapsed = time.time() - t0
        logger.info(f"[FFmpeg/{job_id}] Exit code: {result.returncode} in {elapsed:.1f}s")

        if result.returncode != 0:
            stderr = result.stderr[-3000:]
            logger.error(f"[FFmpeg/{job_id}] STDERR:\n{stderr}")
            raise RuntimeError(f"FFmpeg failed (exit {result.returncode}): {stderr[-500:]}")

        size = os.path.getsize(output_path)
        logger.info(f"[FFmpeg/{job_id}] Output: {size:,} bytes")


# ── Supabase upload ───────────────────────────────────────────────────────────

async def _upload_supabase(local_path: str, job_id: str) -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")

    object_path = f"generated/{job_id}.mp4"
    logger.info(f"[Supabase/{job_id}] Uploading to bucket={SUPABASE_BUCKET} path={object_path}")

    with open(local_path, "rb") as f:
        data = f.read()
    logger.info(f"[Supabase/{job_id}] File size: {len(data):,} bytes")

    def _sync_upload():
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=object_path,
            file=data,
            file_options={"content-type": "video/mp4", "upsert": "true"},
        )

    await asyncio.to_thread(_sync_upload)

    public_url = (
        f"{SUPABASE_URL.rstrip('/')}"
        f"/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    )
    logger.info(f"[Supabase/{job_id}] Upload complete: {public_url}")
    return public_url