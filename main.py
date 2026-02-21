# main.py  — humsafar-video-service
import logging
import subprocess
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import video as video_router

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "Humsafar Video Service",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(video_router.router)


@app.on_event("startup")
async def startup_checks():
    # Check FFmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "no output"
        logger.info(f"[Startup] ✓ FFmpeg found: {first_line}")
    except FileNotFoundError:
        logger.error("[Startup] ✗ FFmpeg NOT FOUND — videos will fail!")
    except Exception as e:
        logger.error(f"[Startup] ✗ FFmpeg check failed: {e}")

    # Check env vars
    import os
    for var in ["OPENROUTER_API_KEY", "SARVAM_API_KEY", "SUPABASE_URL",
                "SUPABASE_SERVICE_KEY", "SUPABASE_BUCKET"]:
        val = os.getenv(var, "")
        if val:
            logger.info(f"[Startup] ✓ {var} is set")
        else:
            logger.error(f"[Startup] ✗ {var} is MISSING")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "humsafar-video-service"}