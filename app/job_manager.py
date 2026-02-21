# app/job_manager.py
# Lightweight in-memory job tracker.
# On Render free tier, memory is cheap compared to FFmpeg CPU.
# Jobs survive within a single dyno lifetime — no Redis needed.

import uuid
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)

JobStatus = Literal["generating", "ready", "failed"]


@dataclass
class Job:
    job_id:    str
    status:    JobStatus   = "generating"
    progress:  int         = 0
    video_url: Optional[str] = None
    message:   str         = ""


class JobManager:
    """Thread-safe (asyncio) in-memory job store."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        job_id = uuid.uuid4().hex[:20]
        job = Job(job_id=job_id)
        self._jobs[job_id] = job
        logger.info(f"[JobManager] Created job_id={job_id}")
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(
        self,
        job_id:    str,
        status:    Optional[JobStatus] = None,
        progress:  Optional[int]       = None,
        video_url: Optional[str]       = None,
        message:   str                 = "",
    ):
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"[JobManager] update called for unknown job_id={job_id}")
            return
        if status   is not None: job.status    = status
        if progress is not None: job.progress  = progress
        if video_url is not None: job.video_url = video_url
        job.message = message
        logger.info(f"[JobManager] job_id={job_id} → {job.status} {job.progress}% {message}")

    def delete(self, job_id: str):
        self._jobs.pop(job_id, None)


# Singleton — imported everywhere
jobs = JobManager()