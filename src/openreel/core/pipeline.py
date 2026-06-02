"""Pipeline orchestrator: video in -> clips out."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

from openreel.config import OpenReelSettings, PresetCriteria
from openreel.core.analyzer import analyze_chunk, split_chunk
from openreel.core.chunking import compute_chunks
from openreel.core.deduplicator import deduplicate
from openreel.core.extractor import extract_all_clips
from openreel.core.video_info import probe_video
from openreel.models import (
    ClipManifest,
    DeduplicatedHighlight,
    ExtractedClip,
    HighlightMoment,
    JobStatus,
    ProgressEvent,
)

logger = logging.getLogger(__name__)


async def process_video(
    input_path: Path,
    settings: OpenReelSettings,
    criteria: str | None = None,
    preset: PresetCriteria | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> AsyncGenerator[ProgressEvent, None]:
    """Main pipeline: analyze a video and extract highlight clips.

    Yields ProgressEvent objects to report status. The final event
    has status COMPLETED or FAILED.

    Args:
        input_path: Path to the source MP4 file.
        settings: Configuration settings.
        criteria: Custom criteria text (overrides preset).
        preset: Preset criteria enum value.
        output_dir: Where to write clips. Defaults to ./openreel_output/<video_stem>.
        dry_run: If True, analyze only — no clip extraction.
    """
    # Resolve output directory (use absolute path so it works regardless of cwd)
    if output_dir is None:
        output_dir = Path("openreel_output") / input_path.stem
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    highlights: list[DeduplicatedHighlight] = []
    clips: list[ExtractedClip] = []
    tmp_dir: Path | None = None

    try:
        # Step 1: Probe video
        yield ProgressEvent(status=JobStatus.ANALYZING, current_step="Probing video metadata...")
        video = await probe_video(input_path)

        # Step 2: Compute chunks
        chunks = compute_chunks(video.duration_seconds, settings, video.file_size_bytes)
        logger.info("Video split into %d analysis chunk(s)", len(chunks))

        # Step 3: Split chunks in parallel (utilize all available CPU cores)
        needs_split = len(chunks) > 1 or video.file_size_bytes > 2 * 1024 * 1024 * 1024
        # Also split if FPS reduction is requested (even single chunk needs re-encoding)
        needs_reencode = settings.analysis_fps > 0 or settings.downscale_for_analysis
        needs_split = needs_split or needs_reencode
        tmp_dir = None
        chunk_files: dict[int, Path] = {}

        if needs_split:
            tmp_dir = Path(tempfile.mkdtemp(prefix="openreel_"))
            yield ProgressEvent(
                status=JobStatus.ANALYZING,
                chunks_total=len(chunks),
                chunks_completed=0,
                current_step=f"Splitting {len(chunks)} chunks in parallel...",
            )

            # Run all chunk splits concurrently
            split_tasks = [
                split_chunk(
                    input_path,
                    chunk_window,
                    tmp_dir,
                    downscale=settings.downscale_for_analysis,
                    analysis_fps=settings.analysis_fps,
                )
                for chunk_window in chunks
            ]
            results = await asyncio.gather(*split_tasks)

            for chunk_window, chunk_file in zip(chunks, results):
                chunk_files[chunk_window.index] = chunk_file
                logger.info(
                    "Split chunk %d/%d: %s (%.1f MB)",
                    chunk_window.index + 1,
                    len(chunks),
                    chunk_file.name,
                    chunk_file.stat().st_size / (1024 * 1024),
                )
        else:
            # Single chunk, use original file directly
            chunk_files[0] = input_path

        yield ProgressEvent(
            status=JobStatus.ANALYZING,
            chunks_total=len(chunks),
            chunks_completed=0,
            current_step=f"Analyzing {len(chunks)} chunk(s) with Gemini...",
        )

        # Step 4: Upload + analyze chunks concurrently (network-bound)
        semaphore = asyncio.Semaphore(settings.max_analysis_concurrency)
        moments_by_chunk: dict[int, list[HighlightMoment]] = {}
        chunks_done = 0

        chunk_errors: list[str] = []

        async def _analyze_chunk(chunk_window):
            async with semaphore:
                try:
                    return chunk_window.index, await analyze_chunk(
                        chunk=chunk_window,
                        chunk_file=chunk_files[chunk_window.index],
                        total_chunks=len(chunks),
                        settings=settings,
                        criteria=criteria,
                        preset=preset,
                        needs_offset=needs_split,
                    )
                except Exception as e:
                    logger.error("Chunk %d failed: %s", chunk_window.index + 1, e)
                    chunk_errors.append(f"Chunk {chunk_window.index + 1}: {e}")
                    return chunk_window.index, []

        # Run all chunk analyses concurrently
        tasks = [_analyze_chunk(c) for c in chunks]
        for coro in asyncio.as_completed(tasks):
            chunk_idx, moments = await coro
            moments_by_chunk[chunk_idx] = moments
            chunks_done += 1

            total_moments = sum(len(m) for m in moments_by_chunk.values())
            error_note = f" ({len(chunk_errors)} failed)" if chunk_errors else ""
            yield ProgressEvent(
                status=JobStatus.ANALYZING,
                chunks_total=len(chunks),
                chunks_completed=chunks_done,
                highlights_found=total_moments,
                current_step=f"Analyzed chunk {chunks_done}/{len(chunks)} — {total_moments} moments found{error_note}",
            )

        # Clean up temp chunk files
        _cleanup_tmp(tmp_dir)

        # If ALL chunks failed, abort
        if len(chunk_errors) == len(chunks):
            yield ProgressEvent(
                status=JobStatus.FAILED,
                current_step="All chunks failed analysis",
                error="; ".join(chunk_errors),
            )
            return

        # Step 4: Deduplicate
        yield ProgressEvent(
            status=JobStatus.ANALYZING,
            chunks_total=len(chunks),
            chunks_completed=len(chunks),
            highlights_found=sum(len(m) for m in moments_by_chunk.values()),
            current_step="Deduplicating highlights...",
        )

        highlights = deduplicate(moments_by_chunk, settings, video.duration_seconds)

        if not highlights:
            logger.warning("No highlights found. Try adjusting criteria or lowering quality threshold.")
            _write_manifest(input_path, video.duration_seconds, settings, highlights, clips, output_dir)
            yield ProgressEvent(
                status=JobStatus.COMPLETED,
                chunks_total=len(chunks),
                chunks_completed=len(chunks),
                highlights_found=0,
                current_step="No highlights found.",
            )
            return

        logger.info("Found %d unique highlights", len(highlights))

        # Step 5: Extract clips (unless dry run)
        if dry_run:
            _write_manifest(input_path, video.duration_seconds, settings, highlights, clips, output_dir)
            yield ProgressEvent(
                status=JobStatus.COMPLETED,
                chunks_total=len(chunks),
                chunks_completed=len(chunks),
                highlights_found=len(highlights),
                clips_total=len(highlights),
                clips_extracted=0,
                current_step=f"Dry run complete. {len(highlights)} highlights written to manifest.",
            )
            return

        yield ProgressEvent(
            status=JobStatus.EXTRACTING,
            chunks_total=len(chunks),
            chunks_completed=len(chunks),
            highlights_found=len(highlights),
            clips_total=len(highlights),
            clips_extracted=0,
            current_step=f"Extracting {len(highlights)} clips...",
        )

        extraction_progress = {"done": 0, "errors": 0}

        def _on_extraction_progress(done: int, errors: int):
            extraction_progress["done"] = done
            extraction_progress["errors"] = errors

        clips, errors = await extract_all_clips(
            input_path=input_path,
            highlights=highlights,
            output_dir=output_dir,
            settings=settings,
            on_progress=_on_extraction_progress,
            video_width=video.width,
            video_height=video.height,
        )

        # Step 6: Write manifest
        _write_manifest(input_path, video.duration_seconds, settings, highlights, clips, output_dir)

        status = JobStatus.COMPLETED
        step_msg = f"Done! Extracted {len(clips)} clips to {output_dir}"
        if errors:
            step_msg += f" ({len(errors)} failed)"

        yield ProgressEvent(
            status=status,
            chunks_total=len(chunks),
            chunks_completed=len(chunks),
            highlights_found=len(highlights),
            clips_total=len(highlights),
            clips_extracted=len(clips),
            current_step=step_msg,
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        _cleanup_tmp(tmp_dir)
        yield ProgressEvent(
            status=JobStatus.FAILED,
            current_step="Pipeline failed",
            error=str(e),
        )


def _cleanup_tmp(tmp_dir: Path | None) -> None:
    """Clean up temporary chunk files."""
    if tmp_dir and tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Cleaned up temp dir: %s", tmp_dir)


def _write_manifest(
    input_path: Path,
    duration: float,
    settings: OpenReelSettings,
    highlights: list[DeduplicatedHighlight],
    clips: list[ExtractedClip],
    output_dir: Path,
) -> None:
    """Write the clip manifest JSON to the output directory."""
    manifest = ClipManifest(
        input_path=str(input_path),
        video_duration_seconds=duration,
        settings_used={
            "gemini_model": settings.gemini_model.value,
            "analysis_mode": settings.analysis_mode.value,
            "target_clips_per_hour": settings.target_clips_per_hour,
            "min_clip_seconds": settings.min_clip_seconds,
            "margin_seconds": settings.margin_seconds,
        },
        highlights=highlights,
        clips=clips,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    logger.info("Manifest written to %s", manifest_path)
