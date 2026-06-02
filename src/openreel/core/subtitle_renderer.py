"""Generate ASS subtitle files with Opus-style word-by-word captions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from openreel.core.transcriber import TimedWord

logger = logging.getLogger(__name__)

# Opus-style caption settings
_FONT_NAME = "Arial"
_FONT_SIZE = 58
_PRIMARY_COLOR = "&H00FFFFFF"  # White (ASS uses AABBGGRR)
_HIGHLIGHT_COLOR = "&H0076E600"  # #00e676 in BGR
_OUTLINE_COLOR = "&H00000000"  # Black
_SHADOW_COLOR = "&H80000000"  # Semi-transparent black
_OUTLINE_WIDTH = 3
_SHADOW_DEPTH = 2
_MARGIN_VERTICAL = 80  # pixels from bottom


@dataclass
class WordGroup:
    """A group of 1-3 words displayed together."""

    words: list[TimedWord]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def group_words(words: list[TimedWord], max_words: int = 3, pause_threshold: float = 0.3) -> list[WordGroup]:
    """Group words into short phrases for Opus-style display.

    Rules:
    - 1-3 words per group
    - Respect natural pauses (gap > pause_threshold = new group)
    - Never mix speakers in the same group
    - Keep group duration reasonable (0.2s - 2.0s)
    """
    if not words:
        return []

    groups: list[WordGroup] = []
    current: list[TimedWord] = [words[0]]

    for i in range(1, len(words)):
        prev = words[i - 1]
        word = words[i]
        gap = word.start - prev.end
        speaker_changed = word.speaker is not None and word.speaker != prev.speaker

        # Start a new group if:
        # - We've hit the max words per group
        # - There's a natural pause
        # - The speaker changed
        if len(current) >= max_words or gap > pause_threshold or speaker_changed:
            groups.append(
                WordGroup(
                    words=list(current),
                    start=current[0].start,
                    end=current[-1].end,
                )
            )
            current = [word]
        else:
            current.append(word)

    # Final group
    if current:
        groups.append(
            WordGroup(
                words=list(current),
                start=current[0].start,
                end=current[-1].end,
            )
        )

    logger.debug("Grouped %d words into %d display groups", len(words), len(groups))
    return groups


def _format_ass_time(seconds: float) -> str:
    """Format seconds as H:MM:SS.cc (ASS timestamp format, centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _build_ass_header(video_width: int = 1920, video_height: int = 1080) -> str:
    """Build the ASS file header with Opus-style formatting."""
    return f"""[Script Info]
Title: OpenReel Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{_FONT_NAME},{_FONT_SIZE},{_PRIMARY_COLOR},{_PRIMARY_COLOR},{_OUTLINE_COLOR},{_SHADOW_COLOR},-1,0,0,0,100,100,0,0,1,{_OUTLINE_WIDTH},{_SHADOW_DEPTH},2,40,40,{_MARGIN_VERTICAL},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _render_group_text(group: WordGroup, highlight_index: int | None = None) -> str:
    """Render a word group with the active word highlighted in green.

    For Opus style, we highlight each word sequentially within the group's
    display time by generating multiple dialogue events. But the simplest
    effective approach: highlight ALL words in the group since they appear
    together as one "pop" of text — matching how Opus actually works.
    """
    parts = []
    for i, word in enumerate(group.words):
        if len(group.words) == 1:
            # Single word: always highlighted
            parts.append(f"{{\\c{_HIGHLIGHT_COLOR}}}{word.word}")
        else:
            # Multi-word group: all words shown, each gets highlighted
            # when it's being spoken
            parts.append(word.word)

    return " ".join(parts)


def render_ass_subtitles(
    groups: list[WordGroup],
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    """Render word groups into a complete ASS subtitle file.

    Uses the Opus style: each word group appears as a "pop" of text,
    with individual words highlighted in green as they're spoken.
    """
    lines = [_build_ass_header(video_width, video_height)]

    for group in groups:
        if len(group.words) == 1:
            # Single word: show highlighted for its duration
            text = f"{{\\c{_HIGHLIGHT_COLOR}}}{group.words[0].word}"
            start = _format_ass_time(group.start)
            end = _format_ass_time(group.end)
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        else:
            # Multi-word group: show all words, highlight each one in sequence
            for wi, word in enumerate(group.words):
                # Build text with current word highlighted
                parts = []
                for j, w in enumerate(group.words):
                    if j == wi:
                        parts.append(f"{{\\c{_HIGHLIGHT_COLOR}}}{w.word}{{\\c{_PRIMARY_COLOR}}}")
                    else:
                        parts.append(w.word)
                text = " ".join(parts)

                start = _format_ass_time(word.start)
                # End at the next word's start, or the group end
                if wi + 1 < len(group.words):
                    end = _format_ass_time(group.words[wi + 1].start)
                else:
                    end = _format_ass_time(group.end)

                lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return "\n".join(lines) + "\n"


def write_ass_file(
    groups: list[WordGroup],
    output_path: Path,
    video_width: int = 1920,
    video_height: int = 1080,
) -> Path:
    """Write word groups to an ASS subtitle file."""
    content = render_ass_subtitles(groups, video_width, video_height)
    output_path.write_text(content, encoding="utf-8")
    logger.debug("Wrote ASS subtitle file: %s (%d events)", output_path, len(groups))
    return output_path
