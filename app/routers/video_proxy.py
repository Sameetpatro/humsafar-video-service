import logging
import os
import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["video-proxy"])

VIDEO_SERVICE_URL = os.getenv("VIDEO_SERVICE_URL", "").rstrip("/")
_PROXY_TIMEOUT    = 30.0

def _video_service_url() -> str:
    if not VIDEO_SERVICE_URL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VIDEO_SERVICE_URL is not configured",
        )
    return VIDEO_SERVICE_URL

class GenerateVideoRequest(BaseModel):
    prompt:        str
    bot_text:      str
    site_id:       str
    site_name:     str
    language_code: str = "en-IN"

class GenerateVideoResponse(BaseModel):
    job_id: str
    status: str

class VideoStatusResponse(BaseModel):
    job_id:    str
    status:    str
    progress:  int        = 0
    video_url: str | None = None
    message:   str        = ""

@router.post("/generate-video", response_model=GenerateVideoResponse)
async def generate_video(req: GenerateVideoRequest):
    base = _video_service_url()
    logger.info(f"[Proxy] POST /generate-video → {base}/generate site={req.site_name}")
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
        try:
            resp = await client.post(f"{base}/generate", json=req.model_dump())
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Video service unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    logger.info(f"[Proxy] job_id={data.get('job_id')} enqueued")
    return GenerateVideoResponse(**data)

@router.get("/video-status/{job_id}", response_model=VideoStatusResponse)
async def video_status(job_id: str):
    base = _video_service_url()
    logger.info(f"[Proxy] GET /video-status/{job_id} → {base}/status/{job_id}")
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
        try:
            resp = await client.get(f"{base}/status/{job_id}")
        except httpx.RequestError as exc:
            logger.warning(f"[Proxy] Video service unreachable: {exc}")
            return VideoStatusResponse(
                job_id=job_id, status="generating", progress=50,
                message="Video service temporarily unavailable, retrying…"
            )
    if resp.status_code in (502, 503):
        logger.warning(f"[Proxy] Video service {resp.status_code}, keeping client polling")
        return VideoStatusResponse(
            job_id=job_id, status="generating", progress=50,
            message="Video service restarting…"
        )
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return VideoStatusResponse(**resp.json())
