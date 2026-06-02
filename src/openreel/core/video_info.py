"""Extract video metadata using ffprobe."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from openreel.exceptions import FFmpegError, VideoFileError
from openreel.models import VideoInfo

logger = logging.getLogger(__name__)


async def probe_video(path: Path) -> VideoInfo:
    """Extract metadata from a video file using ffprobe.

    Raises:
        VideoFileError: If the file doesn't exist or isn't a valid video.
        FFmpegError: If ffprobe fails to execute.
    """
    if not path.exists():
        raise VideoFileError(f"Video file not found: {path}")
    if not path.is_file():
        raise VideoFileError(f"Not a file: {path}")
    supported = {".mp4", ".ts", ".mkv", ".mov", ".avi", ".webm", ".flv"}
    if path.suffix.lower() not in supported:
        raise VideoFileError(f"Unsupported format '{path.suffix}'. Supported: {', '.join(supported)}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        raise FFmpegError("ffprobe not found. Install ffmpeg: https://ffmpeg.org/download.html")

    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed (exit {proc.returncode}): {stderr.decode().strip()}")

    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError as e:
        raise FFmpegError(f"Failed to parse ffprobe output: {e}")

    # Find the video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise VideoFileError(f"No video stream found in {path}")

    duration = float(data.get("format", {}).get("duration", 0))
    if duration <= 0:
        raise VideoFileError(f"Could not determine video duration for {path}")

    file_size = int(data.get("format", {}).get("size", 0)) or path.stat().st_size

    info = VideoInfo(
        path=path,
        duration_seconds=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        codec=video_stream.get("codec_name", "unknown"),
        file_size_bytes=file_size,
    )

    logger.info(
        "Video: %s — %.1f min, %dx%d, %s, %.1f MB",
        path.name,
        info.duration_seconds / 60,
        info.width,
        info.height,
        info.codec,
        info.file_size_bytes / (1024 * 1024),
    )

    return info
