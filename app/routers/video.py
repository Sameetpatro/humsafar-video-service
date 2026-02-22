# app/routers/video.py

import asyncio
import base64
import hashlib
import logging
import os
import subprocess
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET      = os.getenv("SUPABASE_BUCKET", "videos")
SARVAM_API_KEY       = os.getenv("SARVAM_API_KEY", "")

FFMPEG_RESOLUTION = "854:480"
FFMPEG_PRESET     = "ultrafast"
FFMPEG_CRF        = "30"
FFMPEG_TIMEOUT    = 180

TTS_SPEAKER = "ritu"
TTS_MODEL   = "bulbul:v3"

router = APIRouter(tags=["video"])
_ffmpeg_pool = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}


class GenerateVideoRequest(BaseModel):
    bot_text:      str
    site_name:     str
    site_id:       str
    language_code: str = "en-IN"

class GenerateVideoResponse(BaseModel):
    job_id: str
    status: str = "generating"

class VideoStatusResponse(BaseModel):
    job_id:    str
    status:    str
    progress:  int = 0
    video_url: str | None = None
    message:   str = ""


def _make_job_id(bot_text: str, site_id: str) -> str:
    raw = f"{site_id}::{bot_text.strip()[:300]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

def _update_job(job_id: str, **kwargs):
    _jobs.setdefault(job_id, {}).update(kwargs)
    s = _jobs[job_id].get("status", "?")
    p = _jobs[job_id].get("progress", 0)
    m = _jobs[job_id].get("message", "")
    logger.info(f"[Job/{job_id}] {s} {p}% — {m}")


@router.post("/generate", response_model=GenerateVideoResponse)
async def generate_video(req: GenerateVideoRequest, background_tasks: BackgroundTasks):
    logger.info(f"[/generate] site={req.site_name} lang={req.language_code} text_len={len(req.bot_text)}")
    job_id = _make_job_id(req.bot_text, req.site_id)
    existing = _jobs.get(job_id)
    if existing:
        if existing.get("status") == "ready":
            return GenerateVideoResponse(job_id=job_id, status="ready")
        if existing.get("status") == "generating":
            return GenerateVideoResponse(job_id=job_id, status="generating")
    _update_job(job_id, status="generating", progress=0, message="Queued")
    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id, bot_text=req.bot_text,
        site_name=req.site_name, site_id=req.site_id,
        language_code=req.language_code,
    )
    return GenerateVideoResponse(job_id=job_id, status="generating")


@router.get("/status/{job_id}", response_model=VideoStatusResponse)
async def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return VideoStatusResponse(
        job_id=job_id,
        status=job.get("status", "generating"),
        progress=job.get("progress", 0),
        video_url=job.get("video_url"),
        message=job.get("message", ""),
    )


async def _run_pipeline(job_id, bot_text, site_name, site_id, language_code):
    logger.info(f"[Pipeline/{job_id}] ===== STARTING =====")
    t0 = time.time()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    output_path = tmp.name
    try:
        _update_job(job_id, progress=10, message="Generating narration audio…")
        wav_bytes = await _tts(bot_text, language_code)
        _update_job(job_id, progress=30, message="Audio ready")

        _update_job(job_id, progress=35, message="Fetching images…")
        image_paths = await _get_images(site_name, job_id)
        _update_job(job_id, progress=50, message=f"Got {len(image_paths)} images")

        _update_job(job_id, progress=55, message="Rendering video…")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_ffmpeg_pool, _run_ffmpeg_sync, job_id, image_paths, wav_bytes, output_path)

        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg output file missing")

        _update_job(job_id, progress=75, message="Uploading…")
        public_url = await _upload_supabase(output_path, job_id)

        logger.info(f"[Pipeline/{job_id}] ===== COMPLETE in {time.time()-t0:.1f}s =====")
        _update_job(job_id, status="ready", progress=100, video_url=public_url, message="Done")
    except Exception as exc:
        logger.error(f"[Pipeline/{job_id}] FAILED: {exc}", exc_info=True)
        _update_job(job_id, status="failed", progress=0, message=str(exc))
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


async def _tts(text: str, language_code: str) -> bytes:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set")
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "…"
        logger.warning(f"[TTS] Truncated to {len(text)} chars")

    logger.info(f"[TTS] model={TTS_MODEL} speaker={TTS_SPEAKER} lang={language_code} chars={len(text)}")

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
                "speaker":              TTS_SPEAKER,
                "model":                TTS_MODEL,
            },
        )

    logger.info(f"[TTS] Response: {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"[TTS] Error: {resp.text}")
        raise RuntimeError(f"Sarvam TTS error {resp.status_code}: {resp.text[:500]}")

    audios = resp.json().get("audios", [])
    if not audios:
        raise RuntimeError("Sarvam TTS returned empty audio list")
    wav = base64.b64decode(audios[0])
    logger.info(f"[TTS] Success — {len(wav):,} bytes")
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Image fetching — Wikimedia Commons
#
# Strategy (3 attempts, each progressively broader):
#   1. Search "<site_name> India heritage" on Wikimedia Commons
#      → returns actual photos of the monument from Wikipedia articles
#   2. Search "<site_name>" alone
#      → catches cases where "India heritage" narrows results too much
#   3. Google Images open-source fallback via Wikimedia "File:" namespace
#
# Why Wikimedia Commons:
#   - Free, no API key, no rate limit for small traffic
#   - Contains the exact same high-quality photos used in Wikipedia articles
#   - Images are correctly tagged (e.g. "Taj Mahal", "Qutub Minar") — not random
#   - Reliable CDN (upload.wikimedia.org) — no auth, no CORS
# ─────────────────────────────────────────────────────────────────────────────

async def _get_images(site_name: str, job_id: str) -> list[str]:
    """
    Fetch 3–4 contextually relevant images of the heritage site.
    Returns a list of local temp file paths (JPEG).
    Falls back to a placeholder only if every attempt fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"imgs_{job_id}_")

    # Try Wikimedia Commons with progressively simpler queries
    search_queries = [
        f"{site_name} India heritage",
        f"{site_name} India",
        site_name,
    ]

    image_urls: list[str] = []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for query in search_queries:
            if len(image_urls) >= 4:
                break
            urls = await _wikimedia_image_urls(client, query, wanted=4)
            # Extend without duplicates
            for u in urls:
                if u not in image_urls:
                    image_urls.append(u)
            if image_urls:
                logger.info(f"[Images/{job_id}] Wikimedia query '{query}' → {len(image_urls)} URL(s)")

        # Download each URL
        paths: list[str] = []
        for i, url in enumerate(image_urls[:4]):
            dest = os.path.join(tmp_dir, f"img_{i:02d}.jpg")
            ok = await _download_image(client, url, dest)
            if ok:
                paths.append(dest)
                logger.info(f"[Images/{job_id}] Downloaded [{i}]: {url[:80]}")

    if paths:
        return paths

    # Last resort: placeholder
    logger.warning(f"[Images/{job_id}] All Wikimedia attempts failed — using placeholder")
    placeholder = os.path.join(tmp_dir, "placeholder.jpg")
    _write_placeholder_jpg(placeholder)
    return [placeholder]


async def _wikimedia_image_urls(
    client: httpx.AsyncClient,
    query:  str,
    wanted: int = 4
) -> list[str]:
    """
    Query the Wikimedia Commons API for images matching `query`.
    Returns up to `wanted` direct image URLs (JPEG/JPG only).

    API used: MediaWiki action=query, generator=search, prop=imageinfo
    This is the same API Wikipedia itself uses — no key required.
    """
    encoded_query = urllib.parse.quote(query)
    api_url = (
        "https://commons.wikimedia.org/w/api.php"
        "?action=query"
        "&format=json"
        f"&generator=search"
        f"&gsrsearch=File:{encoded_query}"   # restrict to File: namespace (images)
        f"&gsrnamespace=6"                    # namespace 6 = File
        f"&gsrlimit={wanted * 3}"            # fetch 3× wanted to allow filtering
        "&prop=imageinfo"
        "&iiprop=url|mime|size"
        "&iiurlwidth=854"                     # request 854px wide thumbnail — matches video width
        "&origin=*"
    )

    try:
        resp = await client.get(api_url, timeout=15.0)
        if resp.status_code != 200:
            logger.warning(f"[Wikimedia] HTTP {resp.status_code} for query '{query}'")
            return []

        data  = resp.json()
        pages = data.get("query", {}).get("pages", {})

        urls: list[str] = []
        for page in pages.values():
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]

            # Skip non-JPEG (SVG diagrams, maps, logos, audio files)
            mime = info.get("mime", "")
            if mime not in ("image/jpeg", "image/jpg"):
                continue

            # Skip very small images (icons, thumbnails < 200px wide)
            width = info.get("width", 0)
            if width < 300:
                continue

            # Prefer the resized thumbnail URL (854px) if available
            thumb_url = info.get("thumburl", "")
            orig_url  = info.get("url", "")
            url = thumb_url if thumb_url else orig_url
            if url:
                urls.append(url)
            if len(urls) >= wanted:
                break

        logger.debug(f"[Wikimedia] '{query}' → {len(urls)} JPEG URL(s)")
        return urls

    except Exception as e:
        logger.warning(f"[Wikimedia] Exception for query '{query}': {e}")
        return []


async def _download_image(client: httpx.AsyncClient, url: str, dest: str) -> bool:
    """Download a single image URL to dest. Returns True on success."""
    try:
        r = await client.get(url, timeout=15.0)
        if r.status_code == 200 and len(r.content) > 1024:   # >1KB = real image
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
        logger.warning(f"[Download] Bad response {r.status_code} or tiny payload ({len(r.content)}B) for {url[:80]}")
        return False
    except Exception as e:
        logger.warning(f"[Download] Failed {url[:80]}: {e}")
        return False


def _write_placeholder_jpg(path: str):
    with open(path, "wb") as f:
        f.write(bytes([
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
        ]))


def _run_ffmpeg_sync(job_id, image_paths, audio_bytes, output_path):
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
                capture_output=True, text=True, timeout=15
            )
            duration = float(probe.stdout.strip())
        except Exception as e:
            logger.warning(f"[FFmpeg/{job_id}] ffprobe failed: {e}, using 20s")
            duration = 20.0

        dur_per_img = max(duration / len(image_paths), 2.0)
        concat = os.path.join(tmp, "concat.txt")
        with open(concat, "w") as f:
            for p in image_paths:
                f.write(f"file '{p}'\n")
                f.write(f"duration {dur_per_img:.2f}\n")
            f.write(f"file '{image_paths[-1]}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat, "-i", audio_path,
            "-vf", f"scale={FFMPEG_RESOLUTION}:force_original_aspect_ratio=decrease,pad={FFMPEG_RESOLUTION}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF,
            "-c:a", "aac", "-b:a", "96k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", "-r", "24",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")
        logger.info(f"[FFmpeg/{job_id}] Done: {os.path.getsize(output_path):,} bytes")


async def _upload_supabase(local_path: str, job_id: str) -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    object_path = f"generated/{job_id}.mp4"
    with open(local_path, "rb") as f:
        data = f.read()
    def _sync():
        from supabase import create_client
        create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY).storage \
            .from_(SUPABASE_BUCKET).upload(
                path=object_path, file=data,
                file_options={"content-type": "video/mp4", "upsert": "true"},
            )
    await asyncio.to_thread(_sync)
    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    logger.info(f"[Supabase/{job_id}] Done: {url}")
    return url