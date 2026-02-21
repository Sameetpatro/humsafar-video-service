# main.py  — humsafar-video-service
# Standalone FastAPI microservice for video generation.
# Deployed as a separate Render Web Service.

import logging
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

app = FastAPI(
    title       = "Humsafar Video Service",
    version     = "1.0.0",
    description = "Standalone microservice: narration → FFmpeg → Supabase",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(video_router.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "humsafar-video-service"}