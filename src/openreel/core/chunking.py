"""Split a video timeline into overlapping analysis windows."""

from __future__ import annotations

from openreel.config import OpenReelSettings
from openreel.models import ChunkWindow

# Gemini File API upload limit
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def compute_chunks(
    duration_seconds: float,
    settings: OpenReelSettings,
    file_size_bytes: int = 0,
) -> list[ChunkWindow]:
    """Compute analysis chunk windows for a video of the given duration.

    Each chunk fits within both Gemini's context window AND the 2 GB file
    upload limit. Adjacent chunks overlap by `settings.overlap_seconds`
    so highlights near boundaries aren't missed.

    Returns a list of ChunkWindow objects with absolute timestamps.
    """
    max_chunk_by_tokens = settings.max_chunk_seconds
    overlap = settings.overlap_seconds

    # Also constrain by file size: estimate bytes/second and cap chunk duration
    if file_size_bytes > 0 and duration_seconds > 0:
        bytes_per_second = file_size_bytes / duration_seconds
        # Leave 5% headroom below the 2 GB limit
        max_chunk_by_size = (_MAX_UPLOAD_BYTES * 0.95) / bytes_per_second
        max_chunk = min(max_chunk_by_tokens, max_chunk_by_size)
    else:
        max_chunk = max_chunk_by_tokens

    # If the video fits in a single chunk, no splitting needed
    if duration_seconds <= max_chunk:
        return [
            ChunkWindow(
                index=0,
                start_seconds=0.0,
                end_seconds=duration_seconds,
                overlap_before_seconds=0.0,
                overlap_after_seconds=0.0,
            )
        ]

    chunks: list[ChunkWindow] = []
    step = max_chunk - overlap
    start = 0.0
    index = 0

    while start < duration_seconds:
        end = min(start + max_chunk, duration_seconds)

        overlap_before = overlap if index > 0 else 0.0
        overlap_after = overlap if end < duration_seconds else 0.0

        chunks.append(
            ChunkWindow(
                index=index,
                start_seconds=start,
                end_seconds=end,
                overlap_before_seconds=overlap_before,
                overlap_after_seconds=overlap_after,
            )
        )

        start += step
        index += 1

        # Avoid a tiny trailing chunk — absorb it into the last chunk
        if duration_seconds - start < overlap and start < duration_seconds:
            chunks[-1] = chunks[-1].model_copy(
                update={
                    "end_seconds": duration_seconds,
                    "overlap_after_seconds": 0.0,
                }
            )
            break

    return chunks
