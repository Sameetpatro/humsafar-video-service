# video_service/main.py
# STANDALONE video microservice — runs independently from the main backend
# Deploy this separately on Render as its own web service

import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Force stdout logging so Render captures everything
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.info("=== VIDEO SERVICE STARTING ===")
logger.info(f"Python: {sys.version}")
logger.info(f"SUPABASE_URL set: {bool(os.getenv('SUPABASE_URL'))}")
logger.info(f"SUPABASE_SERVICE_KEY set: {bool(os.getenv('SUPABASE_SERVICE_KEY'))}")
logger.info(f"SARVAM_API_KEY set: {bool(os.getenv('SARVAM_API_KEY'))}")
logger.info(f"OPENROUTER_API_KEY set: {bool(os.getenv('OPENROUTER_API_KEY'))}")

# Check ffmpeg is available
import subprocess
try:
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
    logger.info(f"FFmpeg available: {r.stdout.splitlines()[0]}")
except Exception as e:
    logger.error(f"FFmpeg NOT available: {e}")

from fastapi import FastAPI
from app.routers import video as video_router

app = FastAPI(title="Humsafar Video Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router.router)

@app.get("/health")
async def health():
    logger.info("[health] ping")
    return {"status": "ok", "service": "humsafar-video"}

@app.get("/")
async def root():
    return {"service": "Humsafar Video Service", "status": "running"}