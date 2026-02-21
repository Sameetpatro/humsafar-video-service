# app/services/ffmpeg_pipeline.py
# Synchronous FFmpeg runner — always called via asyncio.to_thread.
# Output: 480p (854×480), ultrafast preset, crf=28, +faststart.
# Temp files live in a TemporaryDirectory that is deleted on exit — nothing
# is written to a permanent local path.

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 480p settings ─────────────────────────────────────────────────────────────
RESOLUTION = "854:480"
PRESET     = "ultrafast"
CRF        = "28"


def _create_placeholder_jpeg(path: str):
    """Minimal valid 1×1 black JPEG — zero dependencies."""
    minimal_black_jpeg = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xF5, 0x7F, 0xFF, 0xD9,
    ])
    with open(path, "wb") as f:
        f.write(minimal_black_jpeg)


def _resolve_images(site_id: str) -> list[str]:
    """
    Return absolute paths to JPEG images for this site.
    Falls back to an embedded 1×1 black placeholder — never fails.
    """
    images_dir = Path("static/images") / site_id
    if images_dir.exists():
        paths = sorted(images_dir.glob("*.jpg"))[:5]
        if paths:
            abs_paths = [str(p.resolve()) for p in paths]
            logger.info(f"[FFmpeg] {len(abs_paths)} image(s) from {images_dir}")
            return abs_paths

    placeholder_dir = Path("static/images/_placeholder")
    placeholder_dir.mkdir(parents=True, exist_ok=True)
    placeholder = placeholder_dir / "blank.jpg"
    if not placeholder.exists():
        _create_placeholder_jpeg(str(placeholder))

    logger.info("[FFmpeg] Using placeholder image")
    return [str(placeholder.resolve())]


def run_ffmpeg(audio_bytes: bytes, site_id: str, output_path: str) -> None:
    """
    Synchronous: compose images + audio → 480p MP4 at output_path.

    Everything except output_path is written inside a TemporaryDirectory
    that is cleaned up automatically on return or exception.

    output_path is managed by the caller (background task), which deletes
    it after the Supabase upload succeeds.
    """
    image_paths = _resolve_images(site_id)

    with tempfile.TemporaryDirectory() as tmp:
        # Write audio
        audio_path = os.path.join(tmp, "narration.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Probe audio duration
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "csv=p=0", audio_path],
                capture_output=True, text=True, timeout=15,
            )
            total_duration = float(probe.stdout.strip())
        except Exception as e:
            logger.warning(f"[FFmpeg] ffprobe failed ({e}), defaulting to 30 s")
            total_duration = 30.0

        num_images = len(image_paths)
        dur_per_img = max(total_duration / num_images, 2.0)

        # Concat list — image_paths are absolute, so -safe 0 is sufficient
        concat_path = os.path.join(tmp, "images.txt")
        with open(concat_path, "w") as f:
            for p in image_paths:
                f.write(f"file '{p}'\n")
                f.write(f"duration {dur_per_img:.2f}\n")
            f.write(f"file '{image_paths[-1]}'\n")  # FFmpeg concat sentinel

        logger.info(
            f"[FFmpeg] {num_images} image(s), "
            f"duration={total_duration:.1f}s, output={output_path}"
        )

        # ── FFmpeg command — 480p, ultrafast, crf=28, +faststart ─────────────
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            "-i", audio_path,
            "-vf", (
                f"scale={RESOLUTION}:force_original_aspect_ratio=decrease,"
                f"pad={RESOLUTION}:(ow-iw)/2:(oh-ih)/2:black,"
                "setsar=1"
            ),
            "-c:v",  "libx264",
            "-preset", PRESET,
            "-crf",    CRF,
            "-c:a",  "aac", "-b:a", "96k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            "-r", "24",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            tail = result.stderr[-2000:]
            logger.error(f"[FFmpeg] FAILED (exit {result.returncode}):\n{tail}")
            raise RuntimeError(f"FFmpeg exit {result.returncode}:\n{tail}")

        size = os.path.getsize(output_path)
        logger.info(f"[FFmpeg] ✓ {output_path} ({size:,} bytes)")