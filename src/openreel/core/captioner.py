"""Orchestrates captioning: transcribe → group → render frames → composite."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openreel.core.profanity import censor_text
from openreel.core.subtitle_renderer import WordGroup, group_words
from openreel.core.transcriber import TranscriptionError, transcribe
from openreel.models import CaptionStyle

logger = logging.getLogger(__name__)

# Default speaker colors — used when diarize=True and no custom colors provided
_SPEAKER_COLORS_DEFAULT = [
    (0, 230, 118),  # green (#00e676)
    (100, 181, 246),  # blue (#64b5f6)
    (255, 183, 77),  # orange (#ffb74d)
    (206, 147, 216),  # purple (#ce93d8)
    (240, 98, 146),  # pink (#f06292)
    (77, 208, 225),  # cyan (#4dd0e1)
    (255, 241, 118),  # yellow (#fff176)
    (129, 199, 132),  # light green (#81c784)
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _build_speaker_color_map(
    groups: list[WordGroup],
    custom_colors: dict[str, str] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Assign a unique color to each speaker found in the word groups."""
    speakers: list[str] = []
    for group in groups:
        for word in group.words:
            if word.speaker and word.speaker not in speakers:
                speakers.append(word.speaker)

    color_map = {}
    for i, speaker in enumerate(speakers):
        if custom_colors and speaker in custom_colors:
            color_map[speaker] = _hex_to_rgb(custom_colors[speaker])
        else:
            color_map[speaker] = _SPEAKER_COLORS_DEFAULT[i % len(_SPEAKER_COLORS_DEFAULT)]

    if color_map:
        logger.info(
            "Speaker colors: %s",
            {s: f"#{r:02x}{g:02x}{b:02x}" for s, (r, g, b) in color_map.items()},
        )

    return color_map


def _load_font(font_family: str, font_size: int):
    """Load a font by family name, falling back to system defaults."""
    # Try the requested font across all platforms
    import os

    from PIL import ImageFont

    search_paths = [
        # macOS
        f"/System/Library/Fonts/{font_family}.ttc",
        f"/System/Library/Fonts/{font_family}.ttf",
        f"/Library/Fonts/{font_family}.ttf",
        f"/Library/Fonts/{font_family}.ttc",
        # Linux
        f"/usr/share/fonts/truetype/dejavu/{font_family}.ttf",
        f"/usr/share/fonts/TTF/{font_family}.ttf",
        f"/usr/share/fonts/{font_family}.ttf",
        # Windows
        f"C:\\Windows\\Fonts\\{font_family}.ttf",
        f"C:\\Windows\\Fonts\\{font_family}.ttc",
        os.path.expandvars(f"%LOCALAPPDATA%\\Microsoft\\Windows\\Fonts\\{font_family}.ttf"),
    ]

    # Also try the path directly (if user passes an absolute path)
    if "/" in font_family:
        search_paths.insert(0, font_family)

    for path in search_paths:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue

    # Fallback to system defaults across all platforms
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def _render_caption_frames(
    groups: list[WordGroup],
    video_width: int,
    video_height: int,
    fps: float,
    total_duration: float,
    style: CaptionStyle,
) -> list[tuple[float, bytes]]:
    """Render caption overlay frames as raw RGBA bytes using Pillow."""
    from PIL import Image, ImageDraw

    font_size = max(24, int(video_height * style.font_size_ratio))
    font = _load_font(style.font_family, font_size)

    text_color = _hex_to_rgb(style.text_color)
    highlight_color = _hex_to_rgb(style.highlight_color)
    shadow_color = _hex_to_rgb(style.shadow_color)

    # Build speaker color map
    speaker_colors = _build_speaker_color_map(
        groups,
        custom_colors=style.speaker_colors if style.diarize else None,
    )
    has_speakers = len(speaker_colors) > 0

    # Build timeline events: (start, end, word_texts, highlight_idx, speaker_labels)
    events: list[tuple[float, float, list[str], int, list[str | None]]] = []

    for group in groups:
        speakers = [w.speaker for w in group.words]
        if len(group.words) == 1:
            events.append((group.start, group.end, [group.words[0].word], 0, speakers))
        else:
            for wi, word in enumerate(group.words):
                end = group.words[wi + 1].start if wi + 1 < len(group.words) else group.end
                word_texts = [w.word for w in group.words]
                events.append((word.start, end, word_texts, wi, speakers))

    # Generate overlay frames — one per event transition
    frames: list[tuple[float, bytes]] = []
    blank = Image.new("RGBA", (video_width, video_height), (0, 0, 0, 0))
    blank_bytes = blank.tobytes()

    prev_event_idx = -1
    frame_interval = 1.0 / fps

    t = 0.0
    while t <= total_duration:
        active = None
        active_idx = -1
        for ei, ev in enumerate(events):
            if ev[0] <= t < ev[1]:
                active = ev
                active_idx = ei
                break

        if active_idx != prev_event_idx:
            if active is None:
                frames.append((t, blank_bytes))
            else:
                _, _, word_texts, highlight_idx, speakers = active
                img = Image.new("RGBA", (video_width, video_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                parts: list[tuple[str, float, bool, tuple[int, int, int]]] = []
                total_w = 0.0
                space_w = draw.textlength(" ", font=font)
                for i, word in enumerate(word_texts):
                    w = draw.textlength(word, font=font)
                    is_active = i == highlight_idx
                    if has_speakers and i < len(speakers) and speakers[i]:
                        color = speaker_colors.get(speakers[i], highlight_color)
                    elif is_active:
                        color = highlight_color
                    else:
                        color = text_color
                    parts.append((word, w, is_active, color))
                    total_w += w
                    if i < len(word_texts) - 1:
                        total_w += space_w

                x = (video_width - total_w) / 2
                y = int(video_height * style.position)

                # Draw shadow
                for offset in [(2, 2), (-2, 2), (2, -2), (-2, -2), (0, 3), (3, 0)]:
                    sx = x
                    for word, w, _, _ in parts:
                        draw.text(
                            (sx + offset[0], y + offset[1]),
                            word,
                            font=font,
                            fill=shadow_color + (200,),
                        )
                        sx += w + space_w

                # Draw words
                cx = x
                for word, w, is_active, color in parts:
                    if is_active or has_speakers:
                        fill = color + (255,)
                    else:
                        fill = text_color + (255,)
                    draw.text((cx, y), word, font=font, fill=fill)
                    cx += w + space_w

                frames.append((t, img.tobytes()))
            prev_event_idx = active_idx

        t += frame_interval

    return frames


async def _probe_clip(clip_path: Path) -> tuple[float, float, int, int]:
    """Probe a clip for duration, fps, width, height."""
    import json

    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(clip_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    data = json.loads(stdout.decode())
    duration = float(data["format"]["duration"])

    fps, width, height = 30.0, 1920, 1080
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            r = stream.get("r_frame_rate", "30/1")
            num, den = r.split("/")
            fps = float(num) / float(den)
            width = int(stream.get("width", width))
            height = int(stream.get("height", height))
            break

    return duration, fps, width, height


async def add_captions(
    clip_path: Path,
    style: CaptionStyle | None = None,
    model_size: str = "base",
    openai_api_key: str = "",
    diarize: bool = False,
    hf_token: str = "",
    video_width: int = 1920,
    video_height: int = 1080,
    transcript_cache_path: Path | None = None,
) -> Path:
    """Add styled captions to a clip.

    If style is provided, uses its settings. Otherwise falls back to
    individual parameters for backward compatibility with CLI.

    Returns the path to the captioned clip (same as input).
    """
    if style is None:
        style = CaptionStyle(
            enabled=True,
            highlight_color="#00e676",
            diarize=diarize,
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="openreel_cap_"))
    captioned_path = tmp_dir / "captioned.mp4"

    try:
        # Step 1: Transcribe (uses cache if available)
        words = await transcribe(
            clip_path,
            model_size=model_size,
            openai_api_key=openai_api_key,
            diarize=style.diarize,
            hf_token=hf_token,
            cache_path=transcript_cache_path,
        )

        if not words:
            logger.warning("No speech detected in %s, skipping captions", clip_path.name)
            return clip_path

        # Filter out disabled speakers
        if style.disabled_speakers:
            words = [w for w in words if w.speaker not in style.disabled_speakers]
            logger.info("Filtered out %d disabled speaker(s)", len(style.disabled_speakers))

        # Apply profanity censor
        if style.censor_profanity:
            for w in words:
                w.word = censor_text(w.word)

        if not words:
            logger.warning("All words filtered out, skipping captions")
            return clip_path

        # Step 2: Group words
        groups = group_words(words)
        if not groups:
            logger.warning("No word groups generated for %s, skipping captions", clip_path.name)
            return clip_path

        # Step 3: Probe clip
        duration, fps, video_width, video_height = await _probe_clip(clip_path)

        logger.info(
            "Burning captions into %s (%d word groups, %dx%d @ %.1ffps)...",
            clip_path.name,
            len(groups),
            video_width,
            video_height,
            fps,
        )

        # Step 4: Render caption overlay frames
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            None,
            _render_caption_frames,
            groups,
            video_width,
            video_height,
            fps,
            duration,
            style,
        )

        # Step 5: Write raw frames + encode overlay
        overlay_path = tmp_dir / "overlay.mov"
        raw_path = tmp_dir / "overlay.raw"

        with open(raw_path, "wb") as f:
            frame_map = {round(t * fps): data for t, data in frames}
            total_frames = int(duration * fps)
            current_data = frames[0][1] if frames else None
            for fi in range(total_frames):
                if fi in frame_map:
                    current_data = frame_map[fi]
                if current_data:
                    f.write(current_data)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{video_width}x{video_height}",
            "-r",
            str(fps),
            "-i",
            str(raw_path),
            "-c:v",
            "png",
            str(overlay_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Overlay encode failed: %s", stderr.decode()[-200:])
            return clip_path

        raw_path.unlink(missing_ok=True)

        # Step 6: Composite
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(clip_path),
            "-i",
            str(overlay_path),
            "-filter_complex",
            "overlay=0:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            str(captioned_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Caption composite failed: %s", stderr.decode()[-200:])
            return clip_path

        captioned_path.replace(clip_path)
        logger.info("Captions added to %s", clip_path.name)
        return clip_path

    except TranscriptionError as e:
        logger.error("Transcription failed for %s: %s", clip_path.name, e)
        return clip_path

    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
