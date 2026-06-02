"""Cloud Run Job worker — runs a single video processing job to completion.

Reads configuration from environment variables, downloads the video from GCS
or a URL (Twitch/YouTube via yt-dlp), runs the pipeline, uploads clips to GCS,
and writes results to Supabase.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("openreel.worker")

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for GCS streaming upload

# Use GCS FUSE mount for temp files (disk-backed, not RAM)
# Falls back to /tmp if mount not available (local dev)
_FUSE_MOUNT = "/mnt/scratch"
if os.path.isdir(_FUSE_MOUNT):
    SCRATCH_DIR = f"{_FUSE_MOUNT}/tmp"
    os.makedirs(SCRATCH_DIR, exist_ok=True)
else:
    SCRATCH_DIR = None


def download_from_gcs(gcs_uri: str) -> Path:
    """Download a gs:// URI to a local temp file."""
    from google.cloud import storage as gcs

    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, object_name = parts[0], parts[1]

    client = gcs.Client()
    blob = client.bucket(bucket_name).blob(object_name)

    suffix = Path(object_name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="openreel_dl_", dir=SCRATCH_DIR)
    blob.download_to_filename(tmp.name)
    logger.info("Downloaded %s to %s (%.1f MB)", gcs_uri, tmp.name, Path(tmp.name).stat().st_size / 1e6)
    return Path(tmp.name)


def download_url_to_gcs(url: str, job_id: str, update_fn=None) -> str:
    """Download video from URL via yt-dlp, then upload to GCS.

    Downloads to a temp file (uses disk, not RAM), then streams the file
    to GCS in chunks. The temp file is deleted immediately after upload.
    """
    from google.cloud import storage as gcs

    bucket_name = os.environ.get("GCS_BUCKET", "openreel-uploads")
    object_name = f"downloads/{job_id}/video.mp4"

    # Use /tmp (RAM) for the final output since ffmpeg merge needs random writes.
    # GCS FUSE only supports sequential writes.
    # Download fragments go to SCRATCH_DIR (GCS FUSE) to save RAM.
    output_dir = tempfile.mkdtemp(prefix="openreel_ytdl_")
    output_path = Path(output_dir) / "video.mp4"

    # Prefer pre-muxed formats (no merge needed). Fall back to separate streams.
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-f",
        "best[height<=720][ext=mp4]/best[height<=720]/bestvideo[height<=720]+bestaudio/best",
        "--no-part",
        "-o",
        str(output_path),
        url,
    ]

    # Use GCS FUSE for fragment cache (large sequential writes)
    if SCRATCH_DIR:
        cmd.extend(["--paths", f"temp:{SCRATCH_DIR}"])

    logger.info("Downloading from URL: %s", url)

    # Run yt-dlp with live progress logging
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_log = ""
    for line in proc.stdout:
        line = line.strip()
        if line and "[download]" in line and "%" in line:
            # Log progress periodically
            pct = line.split("%")[0].split()[-1] if "%" in line else ""
            if pct and pct != last_log:
                logger.info("yt-dlp: %s", line)
                last_log = pct
                if update_fn and ("10." in pct or "25." in pct or "50." in pct or "75." in pct or "90." in pct):
                    update_fn({"progress": {"current_step": f"Downloading video... {pct}%"}})

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("yt-dlp download failed")

    if not output_path.exists():
        raise RuntimeError("Download completed but output file not found")

    file_size = output_path.stat().st_size
    logger.info("Downloaded %.1f MB to disk, uploading to GCS...", file_size / 1e6)

    if update_fn:
        update_fn({"progress": {"current_step": "Uploading video to cloud storage..."}})

    # Stream upload to GCS (upload_from_filename reads in chunks, not all at once)
    client = gcs.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    blob.upload_from_filename(str(output_path), content_type="video/mp4", timeout=7200)

    output_path.unlink(missing_ok=True)
    logger.info("Uploaded to gs://%s/%s", bucket_name, object_name)
    return f"gs://{bucket_name}/{object_name}"


def is_url(path: str) -> bool:
    """Check if the input path is a URL (not a GCS path or local file)."""
    return path.startswith("http://") or path.startswith("https://")


def _generate_clip_thumbnail(clip_path: Path) -> Path | None:
    """Extract a thumbnail frame from a clip at 25% in."""
    thumb_path = clip_path.parent / f".thumb_{clip_path.stem}.jpg"
    try:
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 5.0
        seek = duration * 0.25

        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek), "-i", str(clip_path), "-frames:v", "1", "-q:v", "5", str(thumb_path)],
            capture_output=True,
            timeout=15,
        )
        return thumb_path if thumb_path.exists() else None
    except Exception:
        return None


def upload_clips_to_gcs(job_id: str, output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Upload clips + thumbnails to GCS. Returns (clip_paths, thumb_urls)."""
    from google.cloud import storage as gcs

    bucket_name = os.environ.get("GCS_BUCKET", "openreel-uploads")
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    uploaded = {}
    thumbnails = {}

    for clip_file in output_dir.glob("clip_*.mp4"):
        # Upload clip
        object_name = f"clips/{job_id}/{clip_file.name}"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(clip_file))
        gcs_path = f"gs://{bucket_name}/{object_name}"
        uploaded[clip_file.name] = gcs_path

        # Generate and upload thumbnail
        thumb = _generate_clip_thumbnail(clip_file)
        if thumb:
            thumb_object = f"clips/{job_id}/thumbs/{clip_file.stem}.jpg"
            thumb_blob = bucket.blob(thumb_object)
            thumb_blob.upload_from_filename(str(thumb))
            # Generate a signed read URL (7 days)
            thumb_url = thumb_blob.generate_signed_url(
                version="v4",
                expiration=7 * 24 * 3600,
                method="GET",
            )
            thumbnails[clip_file.name] = thumb_url
            thumb.unlink(missing_ok=True)

        logger.info("Uploaded %s -> %s", clip_file.name, gcs_path)

    return uploaded, thumbnails


def update_supabase(job_id: str, updates: dict):
    """Update job row in Supabase."""
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    client.table("jobs").update(updates).eq("id", job_id).execute()


async def run_job():
    """Main worker entry point."""
    # Read config from environment
    job_id = os.environ["JOB_ID"]
    input_path = os.environ["INPUT_PATH"]
    gemini_key = os.environ["GEMINI_API_KEY"]
    preset_str = os.environ.get("PRESET", "general")
    criteria = os.environ.get("CRITERIA") or None
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    analysis_mode = os.environ.get("ANALYSIS_MODE", "fast")
    clips_per_hour = float(os.environ.get("CLIPS_PER_HOUR", "2.5"))
    min_clip_seconds = float(os.environ.get("MIN_CLIP_SECONDS", "15"))
    margin_seconds = float(os.environ.get("MARGIN_SECONDS", "3"))

    logger.info("Worker starting job %s: %s (preset=%s, model=%s)", job_id, input_path, preset_str, gemini_model)

    # Update status to analyzing
    update_supabase(job_id, {"status": "analyzing", "progress": {"current_step": "Starting..."}})

    # Download video from GCS, URL, or use local path
    local_input = None
    if input_path.startswith("gs://"):
        update_supabase(job_id, {"progress": {"current_step": "Downloading video from cloud..."}})
        try:
            local_input = download_from_gcs(input_path)
        except Exception as e:
            logger.error("GCS download failed: %s", e)
            update_supabase(
                job_id,
                {
                    "status": "failed",
                    "error": f"Download failed: {e}",
                    "progress": {"current_step": "Failed"},
                },
            )
            return
    elif is_url(input_path):
        update_supabase(job_id, {"progress": {"current_step": "Downloading video from URL..."}})
        try:
            # Download from URL → disk → GCS
            gcs_path = download_url_to_gcs(input_path, job_id, update_fn=lambda u: update_supabase(job_id, u))
        except Exception as e:
            logger.error("URL download failed: %s", e)
            update_supabase(
                job_id,
                {
                    "status": "failed",
                    "error": f"Download failed: {e}",
                    "progress": {"current_step": "Failed"},
                },
            )
            return

        # Now download from GCS to local for pipeline processing
        update_supabase(job_id, {"progress": {"current_step": "Preparing video for analysis..."}})
        try:
            local_input = download_from_gcs(gcs_path)
        except Exception as e:
            logger.error("GCS download failed: %s", e)
            update_supabase(
                job_id,
                {
                    "status": "failed",
                    "error": f"Failed to prepare video: {e}",
                    "progress": {"current_step": "Failed"},
                },
            )
            return

    resolved_input = local_input or Path(input_path)

    # Build settings
    from openreel.config import AnalysisMode, GeminiModel, OpenReelSettings, PresetCriteria

    settings = OpenReelSettings(
        gemini_api_key=gemini_key,
        gemini_model=GeminiModel(gemini_model),
        analysis_mode=AnalysisMode(analysis_mode),
        target_clips_per_hour=clips_per_hour,
        min_clip_seconds=min_clip_seconds,
        margin_seconds=margin_seconds,
    )

    preset = None
    try:
        preset = PresetCriteria(preset_str)
    except ValueError:
        pass

    scratch = SCRATCH_DIR or "/tmp"
    output_dir = Path(f"{scratch}/openreel_output/{job_id}")

    # Run pipeline
    from openreel.core.pipeline import process_video
    from openreel.models import JobStatus

    try:
        async for event in process_video(
            input_path=resolved_input,
            settings=settings,
            criteria=criteria,
            preset=preset,
            output_dir=output_dir,
        ):
            # Sync progress to Supabase
            progress_data = {
                "current_step": event.current_step,
                "chunks_total": event.chunks_total,
                "chunks_completed": event.chunks_completed,
                "highlights_found": event.highlights_found,
                "clips_extracted": event.clips_extracted,
                "clips_total": event.clips_total,
            }

            updates: dict = {"progress": progress_data}

            if event.status == JobStatus.ANALYZING:
                updates["status"] = "analyzing"
            elif event.status == JobStatus.EXTRACTING:
                updates["status"] = "extracting"

            # Sync highlights when available
            if event.status in (JobStatus.EXTRACTING, JobStatus.COMPLETED):
                manifest_path = output_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    updates["highlights"] = manifest.get("highlights", [])

            update_supabase(job_id, updates)

            if event.status == JobStatus.FAILED:
                update_supabase(
                    job_id,
                    {
                        "status": "failed",
                        "error": event.error or "Unknown error",
                        "progress": progress_data,
                    },
                )
                return

        # Upload clips to GCS
        if output_dir.exists():
            update_supabase(job_id, {"progress": {"current_step": "Uploading clips..."}})
            uploaded, thumbnails = upload_clips_to_gcs(job_id, output_dir)

            # Read final manifest for clips metadata
            manifest_path = output_dir / "manifest.json"
            clips_data = []
            highlights_data = []
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                highlights_data = manifest.get("highlights", [])
                for clip in manifest.get("clips", []):
                    clip_name = Path(clip["output_path"]).name
                    if clip_name in uploaded:
                        clip["output_path"] = uploaded[clip_name]
                    if clip_name in thumbnails:
                        clip["thumbnail_url"] = thumbnails[clip_name]
                    clips_data.append(clip)

            update_supabase(
                job_id,
                {
                    "status": "completed",
                    "highlights": highlights_data,
                    "clips": clips_data,
                    "progress": {"current_step": "Complete"},
                    "error": None,
                },
            )
            logger.info("Job %s completed: %d highlights, %d clips", job_id, len(highlights_data), len(clips_data))
        else:
            update_supabase(
                job_id,
                {
                    "status": "completed",
                    "progress": {"current_step": "Complete (no clips)"},
                },
            )

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        update_supabase(
            job_id,
            {
                "status": "failed",
                "error": str(e),
                "progress": {"current_step": "Failed"},
            },
        )
    finally:
        # Clean up temp files
        if local_input and local_input.exists():
            local_input.unlink(missing_ok=True)
        logger.info("Worker finished job %s", job_id)


def main():
    asyncio.run(run_job())


if __name__ == "__main__":
    main()
