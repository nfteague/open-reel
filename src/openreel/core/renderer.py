"""Render a raw clip with aspect ratio crop, styled captions, and intro/outro."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from openreel.config import AspectRatio
from openreel.core.captioner import _probe_clip, add_captions
from openreel.core.extractor import _get_crop_filter
from openreel.models import CaptionStyle, WatermarkSettings

logger = logging.getLogger(__name__)


async def _apply_watermark(
    clip_path: Path,
    output_path: Path,
    watermark: WatermarkSettings,
    video_width: int,
    video_height: int,
) -> None:
    """Overlay a watermark image onto a clip."""
    if not watermark.image_path:
        raise ValueError("watermark.image_path is required")

    # Compute watermark width in pixels and position offsets
    wm_w = max(1, int(video_width * watermark.size_ratio))
    margin = watermark.margin

    positions = {
        "top-left": f"{margin}:{margin}",
        "top-right": f"main_w-overlay_w-{margin}:{margin}",
        "bottom-left": f"{margin}:main_h-overlay_h-{margin}",
        "bottom-right": f"main_w-overlay_w-{margin}:main_h-overlay_h-{margin}",
    }
    pos = positions.get(watermark.position, positions["bottom-right"])

    # Build filter: scale watermark, apply opacity, overlay on main video
    opacity = max(0.0, min(1.0, watermark.opacity))
    filter_complex = f"[1:v]scale={wm_w}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay={pos}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_path),
        "-i",
        str(watermark.image_path),
        "-filter_complex",
        filter_complex,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(output_path),
    ]

    logger.info("Applying watermark (%s, opacity=%.2f, size=%dpx)", watermark.position, opacity, wm_w)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Watermark overlay failed: {stderr.decode()[-300:]}")


async def _concat_with_intro_outro(
    body_path: Path,
    output_path: Path,
    intro_path: Path | None,
    outro_path: Path | None,
    target_width: int,
    target_height: int,
) -> None:
    """Concatenate intro + body + outro using ffmpeg with scale+crop normalization."""
    inputs: list[Path] = []
    if intro_path:
        inputs.append(intro_path)
    inputs.append(body_path)
    if outro_path:
        inputs.append(outro_path)

    if len(inputs) == 1:
        # No intro/outro, just copy
        shutil.copy2(body_path, output_path)
        return

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd.extend(["-i", str(inp)])

    # Build filter_complex: scale+crop each video, resample audio, then concat
    W, H = target_width, target_height
    video_filters = []
    audio_filters = []
    concat_inputs = []

    for i in range(len(inputs)):
        # Scale to fill (force_original_aspect_ratio=increase scales so the
        # entire frame is covered, then crop centers to W x H)
        video_filters.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},setsar=1,fps=30,format=yuv420p[v{i}]"
        )
        # Normalize audio to stereo 48kHz
        audio_filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")

    filter_complex = (
        ";".join(video_filters)
        + ";"
        + ";".join(audio_filters)
        + ";"
        + "".join(concat_inputs)
        + f"concat=n={len(inputs)}:v=1:a=1[v][a]"
    )

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )

    logger.info("Concatenating %d clips (W=%d, H=%d)", len(inputs), W, H)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Concat failed: {stderr.decode()[-300:]}")


async def render_clip(
    raw_clip_path: Path,
    output_path: Path,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    caption_style: CaptionStyle | None = None,
    intro_path: Path | None = None,
    outro_path: Path | None = None,
    watermark: WatermarkSettings | None = None,
    model_size: str = "base",
    openai_api_key: str = "",
    hf_token: str = "",
    transcript_cache_path: Path | None = None,
    on_step: callable | None = None,
) -> Path:
    """Apply aspect ratio crop, styled captions, and intro/outro to a raw clip.

    The raw clip is not modified. The rendered result is written to output_path.

    Returns output_path on success, raises on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="openreel_render_"))

    try:
        # Start with the raw clip
        working_path = tmp_dir / "working.mp4"
        shutil.copy2(raw_clip_path, working_path)

        def _step(msg: str):
            if on_step:
                on_step(msg)

        # Step 1: Apply aspect ratio crop if needed
        _step(f"Cropping to {aspect_ratio.value}...")
        crop_filter = _get_crop_filter(aspect_ratio)
        if crop_filter:
            cropped_path = tmp_dir / "cropped.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(working_path),
                "-vf",
                crop_filter,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "copy",
                str(cropped_path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Crop failed: {stderr.decode()[-200:]}")
            working_path = cropped_path
            logger.info("Cropped to %s", aspect_ratio.value)

        # Step 2: Apply captions if enabled
        if caption_style and caption_style.enabled:
            _step("Transcribing audio...")
            _, _, width, height = await _probe_clip(working_path)
            await add_captions(
                clip_path=working_path,
                style=caption_style,
                model_size=model_size,
                openai_api_key=openai_api_key,
                hf_token=hf_token,
                video_width=width,
                video_height=height,
                transcript_cache_path=transcript_cache_path,
            )
            logger.info("Captions applied")

        # Step 3: Apply watermark if enabled
        if watermark and watermark.enabled and watermark.image_path:
            _step("Applying watermark...")
            _, _, width, height = await _probe_clip(working_path)
            wm_out = tmp_dir / "watermarked.mp4"
            await _apply_watermark(working_path, wm_out, watermark, width, height)
            working_path = wm_out
            logger.info("Watermark applied")

        # Step 4: Concat intro/outro if specified
        if intro_path or outro_path:
            _step("Joining intro/outro...")
            _, _, body_w, body_h = await _probe_clip(working_path)
            concat_path = tmp_dir / "final.mp4"
            await _concat_with_intro_outro(
                body_path=working_path,
                output_path=concat_path,
                intro_path=intro_path,
                outro_path=outro_path,
                target_width=body_w,
                target_height=body_h,
            )
            working_path = concat_path

        # Move final result to output path
        shutil.copy2(working_path, output_path)
        logger.info(
            "Rendered clip: %s (%.1f MB)",
            output_path.name,
            output_path.stat().st_size / (1024 * 1024),
        )
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
