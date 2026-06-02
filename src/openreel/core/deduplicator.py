"""Merge and deduplicate highlights found near chunk boundaries."""

from __future__ import annotations

import logging

from openreel.config import OpenReelSettings
from openreel.models import DeduplicatedHighlight, HighlightMoment

logger = logging.getLogger(__name__)

# Two moments are considered duplicates if their start/end times
# are within this many seconds of each other.
_PROXIMITY_THRESHOLD = 10.0


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute the overlap ratio between two time ranges.

    Returns the overlap duration as a fraction of the shorter range's duration.
    """
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    shorter_duration = min(a_end - a_start, b_end - b_start)
    if shorter_duration <= 0:
        return 0.0

    return overlap_duration / shorter_duration


def _are_duplicates(a: HighlightMoment, b: HighlightMoment) -> bool:
    """Determine if two moments are duplicates that should be merged."""
    # Check time overlap
    if _overlap_ratio(a.start_seconds, a.end_seconds, b.start_seconds, b.end_seconds) > 0.5:
        return True

    # Check proximity
    start_diff = abs(a.start_seconds - b.start_seconds)
    end_diff = abs(a.end_seconds - b.end_seconds)
    if start_diff < _PROXIMITY_THRESHOLD and end_diff < _PROXIMITY_THRESHOLD:
        return True

    return False


def deduplicate(
    moments_by_chunk: dict[int, list[HighlightMoment]],
    settings: OpenReelSettings,
    video_duration: float,
) -> list[DeduplicatedHighlight]:
    """Deduplicate and merge highlights across chunks.

    Takes moments grouped by chunk index, merges duplicates found in
    overlapping regions, and returns a sorted list of unique highlights
    with padding applied.
    """
    # Flatten all moments with their source chunk index
    tagged: list[tuple[int, HighlightMoment]] = []
    for chunk_idx, moments in moments_by_chunk.items():
        for moment in moments:
            tagged.append((chunk_idx, moment))

    # Sort by start time
    tagged.sort(key=lambda x: x[1].start_seconds)

    # Greedy merge: walk through sorted moments and merge duplicates
    merged_groups: list[list[tuple[int, HighlightMoment]]] = []

    for item in tagged:
        if not merged_groups:
            merged_groups.append([item])
            continue

        # Check if this moment is a duplicate of any moment in the last group
        last_group = merged_groups[-1]
        is_dup = any(_are_duplicates(item[1], existing[1]) for existing in last_group)

        if is_dup:
            last_group.append(item)
        else:
            merged_groups.append([item])

    # Build deduplicated highlights
    results: list[DeduplicatedHighlight] = []
    for idx, group in enumerate(merged_groups):
        # Pick the best moment (highest confidence)
        best_chunk, best_moment = max(group, key=lambda x: x[1].confidence)

        # Time range: union of all moments in group
        start = min(m.start_seconds for _, m in group)
        end = max(m.end_seconds for _, m in group)

        # Validate duration
        if (end - start) < settings.min_clip_seconds:
            logger.debug(
                "Discarding highlight '%s' — too short (%.1fs < %.1fs)",
                best_moment.title,
                end - start,
                settings.min_clip_seconds,
            )
            continue

        # Validate timestamps are within video bounds
        if start < 0 or end > video_duration:
            logger.warning(
                "Highlight '%s' has timestamps outside video bounds (%.1f-%.1f), clamping.",
                best_moment.title,
                start,
                end,
            )
            start = max(0.0, start)
            end = min(video_duration, end)

        # Apply padding
        padded_start = max(0.0, start - settings.margin_seconds)
        padded_end = min(video_duration, end + settings.margin_seconds)

        # Collect tags and source chunks
        all_tags = set()
        source_chunks = set()
        for chunk_idx, moment in group:
            all_tags.update(moment.tags)
            source_chunks.add(chunk_idx)

        results.append(
            DeduplicatedHighlight(
                index=idx,
                title=best_moment.title,
                description=best_moment.description,
                start_seconds=start,
                end_seconds=end,
                padded_start_seconds=padded_start,
                padded_end_seconds=padded_end,
                confidence=best_moment.confidence,
                tags=sorted(all_tags),
                source_chunks=sorted(source_chunks),
            )
        )

    # Re-index sequentially
    for i, highlight in enumerate(results):
        highlight.index = i

    logger.info(
        "Deduplication: %d raw moments -> %d unique highlights",
        sum(len(m) for m in moments_by_chunk.values()),
        len(results),
    )

    return results
