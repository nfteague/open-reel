"""REST API endpoints for OpenReel."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from openreel.api.dependencies import get_settings
from openreel.config import AnalysisMode, GeminiModel, OpenReelSettings, PresetCriteria
from openreel.core.pipeline import process_video
from openreel.models import (
    ClipRenderSettings,
    DeduplicatedHighlight,
    ExtractedClip,
    JobRequest,
    JobResponse,
    JobStatus,
    ProgressEvent,
    RenderRequest,
    RenderResponse,
    RenderStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Persistent job storage directory
_JOBS_DIR = Path(".openreel/jobs")


# --- Job State ---


@dataclass
class Job:
    """Job state, persisted to disk as JSON."""

    id: str
    request: JobRequest
    progress: ProgressEvent = field(
        default_factory=lambda: ProgressEvent(status=JobStatus.PENDING, current_step="Queued")
    )
    highlights: list[DeduplicatedHighlight] = field(default_factory=list)
    clips: list[ExtractedClip] = field(default_factory=list)
    task: asyncio.Task | None = None


_jobs: dict[str, Job] = {}


# --- Persistence ---


def _save_job(job: Job) -> None:
    """Persist job state to disk."""
    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _JOBS_DIR / f"{job.id}.json"
    data = {
        "id": job.id,
        "request": job.request.model_dump(),
        "progress": job.progress.model_dump(),
        "highlights": [h.model_dump() for h in job.highlights],
        "clips": [c.model_dump() for c in job.clips],
    }
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_job(path: Path) -> Job | None:
    """Load a job from a persisted JSON file."""
    try:
        data = json.loads(path.read_text())
        job = Job(
            id=data["id"],
            request=JobRequest(**data["request"]),
            progress=ProgressEvent(**data["progress"]),
            highlights=[DeduplicatedHighlight(**h) for h in data.get("highlights", [])],
            clips=[ExtractedClip(**c) for c in data.get("clips", [])],
        )
        # Mark interrupted jobs
        if job.progress.status in (JobStatus.ANALYZING, JobStatus.EXTRACTING, JobStatus.PENDING):
            job.progress = job.progress.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "error": "Interrupted — server was stopped before this job completed.",
                    "current_step": "Interrupted",
                }
            )
            _save_job(job)
        return job
    except Exception as e:
        logger.warning("Failed to load job from %s: %s", path, e)
        return None


def _delete_job_file(job_id: str) -> None:
    """Remove a persisted job file."""
    path = _JOBS_DIR / f"{job_id}.json"
    path.unlink(missing_ok=True)


def load_persisted_jobs() -> None:
    """Load all persisted jobs on startup."""
    if not _JOBS_DIR.exists():
        return
    for path in _JOBS_DIR.glob("*.json"):
        job = _load_job(path)
        if job:
            _jobs[job.id] = job
            logger.info("Loaded job %s (%s)", job.id, job.progress.status.value)


# --- Pipeline Runner ---


def _download_from_gcs(gcs_uri: str) -> Path:
    """Download a gs:// URI to a local temp file. Returns the local path."""
    import tempfile

    from google.cloud import storage as gcs

    # Parse gs://bucket/object
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, object_name = parts[0], parts[1]

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    suffix = Path(object_name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="openreel_dl_")
    blob.download_to_filename(tmp.name)
    logger.info("Downloaded %s to %s (%.1f MB)", gcs_uri, tmp.name, Path(tmp.name).stat().st_size / 1e6)
    return Path(tmp.name)


async def _run_job(job: Job, settings: OpenReelSettings) -> None:
    """Background task that runs the processing pipeline for a job."""
    local_input = None
    try:
        req = job.request

        # Build settings overrides from request
        overrides = {}
        if req.gemini_model:
            try:
                overrides["gemini_model"] = GeminiModel(req.gemini_model)
            except ValueError:
                pass
        if req.analysis_mode:
            try:
                overrides["analysis_mode"] = AnalysisMode(req.analysis_mode)
            except ValueError:
                pass

        if req.api_key:
            overrides["gemini_api_key"] = req.api_key
        if req.openai_api_key:
            overrides["openai_api_key"] = req.openai_api_key

        job_settings = settings.model_copy(
            update={
                **overrides,
                "target_clips_per_hour": req.target_clips_per_hour,
                "min_clip_seconds": req.min_clip_seconds,
                "margin_seconds": req.margin_seconds,
                "accurate_cuts": req.accurate_cuts,
                "captions": req.captions,
                "caption_model_size": req.caption_model_size,
            }
        )

        # Resolve criteria
        criteria = req.criteria
        preset = None
        if req.preset:
            try:
                preset = PresetCriteria(req.preset)
            except ValueError:
                pass

        output_dir = Path(req.output_dir) if req.output_dir else None

        # Download from GCS if needed
        if req.input_path.startswith("gs://"):
            job.progress = ProgressEvent(status=JobStatus.PENDING, current_step="Downloading video from cloud...")
            _save_job(job)
            local_input = await asyncio.to_thread(_download_from_gcs, req.input_path)

        input_path = local_input or Path(req.input_path)

        async for event in process_video(
            input_path=input_path,
            settings=job_settings,
            criteria=criteria,
            preset=preset,
            output_dir=output_dir,
        ):
            job.progress = event

            # Sync highlights from manifest when available
            if event.status in (JobStatus.EXTRACTING, JobStatus.COMPLETED):
                _sync_manifest(job, input_path, output_dir)

            # Persist after every progress update
            _save_job(job)

    except asyncio.CancelledError:
        job.progress = ProgressEvent(status=JobStatus.FAILED, current_step="Cancelled", error="Job was cancelled")
        _save_job(job)
    except Exception as e:
        logger.error("Job %s failed: %s", job.id, e)
        job.progress = ProgressEvent(status=JobStatus.FAILED, current_step="Failed", error=str(e))
        _save_job(job)
    finally:
        # Clean up GCS download temp file
        if local_input and local_input is not None and Path(local_input).exists():
            Path(local_input).unlink(missing_ok=True)
            logger.info("Cleaned up temp file %s", local_input)


def _sync_manifest(job: Job, input_path: Path, output_dir: Path | None) -> None:
    """Read highlights/clips from the manifest file into the job."""
    if output_dir is None:
        output_dir = Path("openreel_output") / input_path.stem

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            job.highlights = [DeduplicatedHighlight(**h) for h in data.get("highlights", [])]
            job.clips = [ExtractedClip(**c) for c in data.get("clips", [])]
        except Exception:
            pass


# --- Response Helpers ---


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=job.progress.status,
        progress=job.progress,
        highlights=job.highlights or None,
        clips=job.clips or None,
        error=job.progress.error,
    )


# --- Endpoints ---


@router.post("/jobs", status_code=202, response_model=JobResponse)
async def create_job(
    request: JobRequest,
    settings: OpenReelSettings = Depends(get_settings),
) -> JobResponse:
    """Submit a new video processing job."""
    # Accept both local paths and gs:// URIs
    if not request.input_path.startswith("gs://"):
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=f"Input file not found: {request.input_path}")

    job_id = str(uuid.uuid4())[:8]
    job = Job(id=job_id, request=request)
    _jobs[job_id] = job
    _save_job(job)

    # Launch background task
    job.task = asyncio.create_task(_run_job(job, settings))

    logger.info("Created job %s for %s", job_id, request.input_path)
    return _job_to_response(job)


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """List all jobs (current and past)."""
    return [_job_to_response(job) for job in _jobs.values()]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Get the status and results of a processing job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _job_to_response(job)


@router.get("/jobs/{job_id}/highlights", response_model=list[DeduplicatedHighlight])
async def get_highlights(job_id: str) -> list[DeduplicatedHighlight]:
    """Get highlights for a job (available during and after analysis)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.highlights


@router.get("/jobs/{job_id}/clips", response_model=list[ExtractedClip])
async def get_clips(job_id: str) -> list[ExtractedClip]:
    """Get extracted clips for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.progress.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not yet completed")
    return job.clips


@router.get("/jobs/{job_id}/clips/{clip_index}/download")
async def download_clip(job_id: str, clip_index: int):
    """Get a signed download URL for a completed clip."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clip = None
    for c in job.clips:
        if c.highlight.index == clip_index:
            clip = c
            break

    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {clip_index} not found")

    clip_path = str(clip.output_path)

    # If clip is on GCS, generate a signed download URL
    if clip_path.startswith("gs://"):
        from google.cloud import storage as gcs

        parts = clip_path.replace("gs://", "").split("/", 1)
        bucket_name, object_name = parts[0], parts[1]

        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=3600,  # 1 hour
            method="GET",
        )
        return {"download_url": url, "filename": Path(object_name).name}

    # Local file fallback
    from fastapi.responses import FileResponse

    local_path = Path(clip_path)
    if not local_path.exists():
        raise HTTPException(status_code=404, detail="Clip file not found")
    return FileResponse(local_path, media_type="video/mp4", filename=local_path.name)


@router.delete("/jobs/{job_id}", status_code=204, response_model=None)
async def delete_job(job_id: str):
    """Cancel a running job or remove a completed one."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.task and not job.task.done():
        job.task.cancel()

    _delete_job_file(job_id)
    del _jobs[job_id]
    logger.info("Deleted job %s", job_id)


@router.get("/asset")
async def serve_asset(path: str):
    """Serve a local file (image or any media) for preview purposes."""
    import mimetypes

    from fastapi.responses import FileResponse

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    mime, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(file_path, media_type=mime or "application/octet-stream")


@router.get("/probe")
async def probe_file(path: str) -> dict:
    """Probe a video file for dimensions and duration."""
    import json

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    if proc.returncode != 0:
        raise HTTPException(status_code=400, detail="Failed to probe file")

    data = json.loads(stdout.decode())
    width, height, duration = 0, 0, 0.0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            break
    duration = float(data.get("format", {}).get("duration", 0))

    return {
        "path": str(file_path),
        "width": width,
        "height": height,
        "duration": duration,
    }


# --- Session-level Defaults ---


@router.get("/jobs/{job_id}/session-defaults", response_model=ClipRenderSettings)
async def get_session_defaults(job_id: str) -> ClipRenderSettings:
    """Get session-wide default render settings."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    input_path = Path(job.request.input_path)
    output_dir = Path(job.request.output_dir) if job.request.output_dir else Path("openreel_output") / input_path.stem
    manifest_path = output_dir / "manifest.json"

    if not manifest_path.exists():
        return ClipRenderSettings()

    try:
        data = json.loads(manifest_path.read_text())
        defaults = data.get("session_defaults") or {}
        return ClipRenderSettings(**defaults)
    except Exception:
        return ClipRenderSettings()


@router.put("/jobs/{job_id}/session-defaults", response_model=ClipRenderSettings)
async def set_session_defaults(job_id: str, settings: ClipRenderSettings) -> ClipRenderSettings:
    """Save session-wide default render settings to the manifest."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    input_path = Path(job.request.input_path)
    output_dir = Path(job.request.output_dir) if job.request.output_dir else Path("openreel_output") / input_path.stem
    manifest_path = output_dir / "manifest.json"

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")

    data = json.loads(manifest_path.read_text())
    data["session_defaults"] = settings.model_dump()
    manifest_path.write_text(json.dumps(data, indent=2, default=str))

    return settings


@router.get("/browse")
async def browse_files(path: str = "~") -> dict:
    """Browse the local filesystem. Returns directories and MP4 files."""
    browse_path = Path(path).expanduser().resolve()

    if not browse_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {browse_path}")
    if not browse_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    items = []
    try:
        for entry in sorted(browse_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                items.append({"name": entry.name, "path": str(entry), "type": "directory"})
            elif entry.suffix.lower() in (".mp4", ".png", ".jpg", ".jpeg", ".webp", ".gif"):
                size_mb = entry.stat().st_size / (1024 * 1024)
                items.append(
                    {
                        "name": entry.name,
                        "path": str(entry),
                        "type": "file",
                        "size_mb": round(size_mb, 1),
                    }
                )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    parent = str(browse_path.parent) if browse_path != browse_path.parent else None

    return {
        "current": str(browse_path),
        "parent": parent,
        "items": items,
    }


@router.get("/health")
async def health_check() -> dict:
    """Check system health: ffmpeg availability and API key configuration."""
    import os
    import shutil

    from dotenv import load_dotenv

    # Mirror how jobs resolve the key (config.resolve_api_key) so a key set only
    # in the project-root .env is reported as configured rather than "missing".
    load_dotenv()

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    api_key_set = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENREEL_GEMINI_API_KEY"))

    # Check if pyannote.audio is installed for speaker diarization
    try:
        import pyannote.audio  # noqa: F401

        diarization_ok = True
    except ImportError:
        diarization_ok = False

    healthy = ffmpeg_ok and ffprobe_ok and api_key_set

    return {
        "healthy": healthy,
        "ffmpeg": "ok" if ffmpeg_ok else "not found",
        "ffprobe": "ok" if ffprobe_ok else "not found",
        "gemini_api_key": "configured" if api_key_set else "missing",
        "diarization": "ok" if diarization_ok else "not installed (optional)",
    }


# --- Thumbnails ---


@router.get("/jobs/{job_id}/thumbnail/{clip_index}")
async def get_thumbnail(job_id: str, clip_index: int):
    """Get a thumbnail frame from a raw clip."""
    from fastapi.responses import FileResponse

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find the clip
    clip = None
    for c in job.clips:
        if c.highlight.index == clip_index:
            clip = c
            break

    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {clip_index} not found")

    clip_path = Path(clip.output_path)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip file not found")

    # Check for cached thumbnail
    thumb_path = clip_path.parent / f".thumb_{clip_path.stem}.jpg"
    if not thumb_path.exists():
        # Extract frame at 25% into the clip
        duration = clip.duration_seconds
        seek_time = duration * 0.25

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-ss",
            str(seek_time),
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(thumb_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if not thumb_path.exists():
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    return FileResponse(thumb_path, media_type="image/jpeg")


# --- Render ---

# In-memory render task tracking
_render_tasks: dict[str, dict] = {}  # "{job_id}:{clip_index}" -> {task, status, output_path, error}


@router.post("/jobs/{job_id}/render/{clip_index}", response_model=RenderResponse)
async def render_clip_endpoint(
    job_id: str,
    clip_index: int,
    request: RenderRequest,
    settings: OpenReelSettings = Depends(get_settings),
):
    """Render a clip with aspect ratio crop and/or captions."""
    from openreel.config import AspectRatio
    from openreel.core.renderer import render_clip

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clip = None
    for c in job.clips:
        if c.highlight.index == clip_index:
            clip = c
            break

    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {clip_index} not found")

    raw_path = Path(clip.output_path)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Raw clip file not found")

    # Determine output path
    ratio_tag = request.aspect_ratio.replace(":", "x")
    output_dir = raw_path.parent / "rendered"
    output_path = output_dir / f"{raw_path.stem}_{ratio_tag}.mp4"

    # Resolve aspect ratio enum
    try:
        aspect = AspectRatio(request.aspect_ratio)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid aspect ratio: {request.aspect_ratio}")

    render_key = f"{job_id}:{clip_index}:{ratio_tag}"

    # Launch render as background task
    async def _do_render():
        try:
            _render_tasks[render_key]["status"] = RenderStatus.RENDERING
            _render_tasks[render_key]["current_step"] = "Starting render..."

            def _on_step(msg: str):
                _render_tasks[render_key]["current_step"] = msg

            await render_clip(
                raw_clip_path=raw_path,
                output_path=output_path,
                aspect_ratio=aspect,
                caption_style=request.caption_style if request.caption_style.enabled else None,
                intro_path=Path(request.intro_path) if request.intro_path else None,
                outro_path=Path(request.outro_path) if request.outro_path else None,
                watermark=request.watermark if request.watermark.enabled else None,
                model_size=settings.caption_model_size,
                openai_api_key=request.openai_api_key or settings.openai_api_key,
                hf_token=request.hf_token or settings.hf_token,
                transcript_cache_path=_transcript_cache_path(job_id, clip_index),
                on_step=_on_step,
            )
            _render_tasks[render_key]["status"] = RenderStatus.COMPLETED
            _render_tasks[render_key]["output_path"] = str(output_path)
            _render_tasks[render_key]["current_step"] = "Done"
        except Exception as e:
            logger.error("Render failed: %s", e)
            _render_tasks[render_key]["status"] = RenderStatus.FAILED
            _render_tasks[render_key]["error"] = str(e)

    _render_tasks[render_key] = {
        "status": RenderStatus.PENDING,
        "current_step": "Queued...",
        "output_path": None,
        "error": None,
        "task": asyncio.create_task(_do_render()),
    }

    # Save settings to manifest
    _save_clip_settings(
        job,
        clip_index,
        ClipRenderSettings(
            aspect_ratio=request.aspect_ratio,
            caption_style=request.caption_style,
            intro_path=request.intro_path,
            outro_path=request.outro_path,
            watermark=request.watermark,
        ),
    )

    return RenderResponse(status=RenderStatus.PENDING)


@router.get("/jobs/{job_id}/render/{clip_index}/status")
async def render_status(job_id: str, clip_index: int, ratio: str = "16:9"):
    """Poll render progress."""
    ratio_tag = ratio.replace(":", "x")
    render_key = f"{job_id}:{clip_index}:{ratio_tag}"

    task_info = _render_tasks.get(render_key)
    if not task_info:
        raise HTTPException(status_code=404, detail="No render in progress")

    return RenderResponse(
        status=task_info["status"],
        current_step=task_info.get("current_step"),
        output_path=task_info["output_path"],
        error=task_info["error"],
    )


@router.get("/jobs/{job_id}/renders/{clip_index}")
async def list_renders(job_id: str, clip_index: int):
    """List previously rendered versions of a clip."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clip = None
    for c in job.clips:
        if c.highlight.index == clip_index:
            clip = c
            break

    if not clip:
        return []

    raw_path = Path(clip.output_path)
    rendered_dir = raw_path.parent / "rendered"
    if not rendered_dir.exists():
        return []

    renders = []
    stem = raw_path.stem
    for f in sorted(rendered_dir.glob(f"{stem}_*.mp4")):
        # Extract ratio tag from filename (e.g., clip_001_..._16x9.mp4 -> 16:9)
        ratio_tag = f.stem[len(stem) + 1 :]
        ratio = ratio_tag.replace("x", ":")
        size_mb = f.stat().st_size / (1024 * 1024)
        renders.append(
            {
                "filename": f.name,
                "path": str(f),
                "ratio": ratio,
                "size_mb": round(size_mb, 1),
            }
        )

    return renders


def _save_clip_settings(job: Job, clip_index: int, settings: ClipRenderSettings) -> None:
    """Save per-clip render settings to the manifest on disk."""
    input_path = Path(job.request.input_path)
    output_dir = Path(job.request.output_dir) if job.request.output_dir else Path("openreel_output") / input_path.stem
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            if "clip_settings" not in data:
                data["clip_settings"] = {}
            data["clip_settings"][str(clip_index)] = settings.model_dump()
            manifest_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save clip settings: %s", e)


# --- Speakers (Transcription) ---

# In-memory transcription tasks
_transcribe_tasks: dict[str, dict] = {}


def _transcript_cache_path(job_id: str, clip_index: int) -> Path:
    """Path for caching transcripts per clip."""
    return Path(".openreel/transcripts") / f"{job_id}_{clip_index}.json"


@router.post("/jobs/{job_id}/clips/{clip_index}/transcribe")
async def transcribe_clip(
    job_id: str,
    clip_index: int,
    diarize: bool = True,
    openai_api_key: str | None = None,
    hf_token: str | None = None,
    settings: OpenReelSettings = Depends(get_settings),
):
    """Transcribe a clip and detect speakers. Returns task ID for polling."""
    from openreel.core.transcriber import summarize_speakers
    from openreel.core.transcriber import transcribe as do_transcribe

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clip = None
    for c in job.clips:
        if c.highlight.index == clip_index:
            clip = c
            break

    if not clip:
        raise HTTPException(status_code=404, detail=f"Clip {clip_index} not found")

    clip_path = Path(clip.output_path)
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip file not found")

    cache_path = _transcript_cache_path(job_id, clip_index)
    task_key = f"{job_id}:{clip_index}:transcribe"

    async def _do_transcribe():
        try:
            _transcribe_tasks[task_key]["status"] = "running"
            words = await do_transcribe(
                clip_path,
                model_size=settings.caption_model_size,
                openai_api_key=openai_api_key or settings.openai_api_key,
                diarize=diarize,
                hf_token=hf_token or settings.hf_token,
                cache_path=cache_path,
            )
            speakers = summarize_speakers(words) if diarize else []
            _transcribe_tasks[task_key]["status"] = "completed"
            _transcribe_tasks[task_key]["speakers"] = speakers
            _transcribe_tasks[task_key]["word_count"] = len(words)
        except Exception as e:
            logger.error("Transcribe failed: %s", e)
            _transcribe_tasks[task_key]["status"] = "failed"
            _transcribe_tasks[task_key]["error"] = str(e)

    _transcribe_tasks[task_key] = {
        "status": "pending",
        "speakers": [],
        "word_count": 0,
        "error": None,
        "task": asyncio.create_task(_do_transcribe()),
    }

    return {"status": "pending"}


@router.get("/jobs/{job_id}/clips/{clip_index}/transcribe/status")
async def transcribe_status(job_id: str, clip_index: int):
    """Get the status of a transcription task."""
    task_key = f"{job_id}:{clip_index}:transcribe"
    info = _transcribe_tasks.get(task_key)
    if not info:
        raise HTTPException(status_code=404, detail="No transcription in progress")
    return {
        "status": info["status"],
        "speakers": info["speakers"],
        "word_count": info["word_count"],
        "error": info.get("error"),
    }


# --- Fonts ---


# Curated list of caption-friendly fonts likely present on macOS / Linux / Windows.
# Each entry has display name + file-name patterns (without extension) to search for.
_CURATED_FONTS = [
    # Sans-serif system fonts (popular for captions)
    ("Helvetica", ["Helvetica", "HelveticaNeue", "Helvetica-Bold"]),
    ("Helvetica Neue", ["HelveticaNeue", "HelveticaNeueBold"]),
    ("Arial", ["Arial", "ArialMT", "arial", "Arial-Bold"]),
    ("Arial Black", ["Arial Black", "ArialBlack", "ariblk"]),
    ("Impact", ["Impact", "impact"]),
    ("Verdana", ["Verdana", "verdana"]),
    ("Tahoma", ["Tahoma", "tahoma"]),
    ("Trebuchet MS", ["Trebuchet MS", "trebuc"]),
    # Apple-specific
    ("SF Pro", ["SFPro", "SFProDisplay", "SF-Pro", "SFProText"]),
    ("Avenir", ["Avenir", "AvenirNext"]),
    ("Futura", ["Futura"]),
    ("Gill Sans", ["GillSans", "Gill Sans"]),
    # Windows-specific
    ("Segoe UI", ["segoeui", "Segoe UI", "segoeuib"]),
    ("Calibri", ["calibri", "Calibri", "calibrib"]),
    # Serif (less common for captions but useful)
    ("Georgia", ["Georgia", "georgia"]),
    ("Times New Roman", ["Times New Roman", "times", "TimesNewRomanPS"]),
    # Monospace
    ("Courier New", ["Courier New", "cour", "CourierNew"]),
    ("Menlo", ["Menlo"]),
    ("Consolas", ["consola", "Consolas"]),
    # Fun / display
    ("Comic Sans MS", ["Comic Sans MS", "comic", "ComicSansMS"]),
    # Linux defaults
    ("DejaVu Sans", ["DejaVuSans", "DejaVuSans-Bold"]),
    ("Liberation Sans", ["LiberationSans"]),
    ("Noto Sans", ["NotoSans"]),
    ("Ubuntu", ["Ubuntu"]),
]


@router.get("/fonts")
async def list_fonts() -> list[dict]:
    """List curated caption-friendly fonts that are actually installed on the system."""
    import os

    font_dirs = [
        # macOS
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        # Linux
        "/usr/share/fonts",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/TTF",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        # Windows
        "C:\\Windows\\Fonts",
        os.path.expandvars("%LOCALAPPDATA%\\Microsoft\\Windows\\Fonts"),
    ]

    # Build an index of available font files (filename stem → full path)
    available: dict[str, str] = {}
    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for root, _, files in os.walk(font_dir):
            for f in files:
                if f.endswith((".ttf", ".ttc", ".otf")):
                    stem = os.path.splitext(f)[0]
                    if stem not in available:
                        available[stem.lower()] = os.path.join(root, f)

    # Filter curated list to only fonts present on this system
    fonts: list[dict] = []
    for display_name, patterns in _CURATED_FONTS:
        for p in patterns:
            if p.lower() in available:
                fonts.append(
                    {
                        "name": display_name,
                        "path": available[p.lower()],
                    }
                )
                break

    return fonts
