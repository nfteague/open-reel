"""Pydantic data models for OpenReel."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# --- Gemini Response Schema ---


class HighlightMoment(BaseModel):
    """A single highlight moment identified by Gemini."""

    title: str = Field(description="Short descriptive title for the highlight")
    description: str = Field(description="Why this moment is highlight-worthy")
    start_seconds: float = Field(description="Start timestamp in seconds from video start")
    end_seconds: float = Field(description="End timestamp in seconds from video start")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    tags: list[str] = Field(description="Categorization tags like 'funny', 'skillful', 'dramatic'")


class ChunkAnalysisResult(BaseModel):
    """Structured output from analyzing a single video chunk."""

    moments: list[HighlightMoment]
    chunk_summary: str = Field(description="Brief summary of what happened in this segment")


# --- Internal Domain Models ---


class ChunkWindow(BaseModel):
    """Defines a time window for analysis."""

    index: int
    start_seconds: float
    end_seconds: float
    overlap_before_seconds: float = 0.0
    overlap_after_seconds: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class VideoInfo(BaseModel):
    """Metadata extracted from ffprobe."""

    path: Path
    duration_seconds: float
    width: int
    height: int
    codec: str
    file_size_bytes: int


class DeduplicatedHighlight(BaseModel):
    """A highlight after deduplication, ready for extraction."""

    index: int
    title: str
    description: str
    start_seconds: float
    end_seconds: float
    padded_start_seconds: float
    padded_end_seconds: float
    confidence: float
    tags: list[str]
    source_chunks: list[int]

    @property
    def duration_seconds(self) -> float:
        return self.padded_end_seconds - self.padded_start_seconds


class ExtractedClip(BaseModel):
    """Result of extracting a single clip."""

    highlight: DeduplicatedHighlight
    output_path: Path
    duration_seconds: float


# --- Job Models (for API) ---


class JobStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressEvent(BaseModel):
    """Progress report during processing."""

    status: JobStatus
    chunks_total: int = 0
    chunks_completed: int = 0
    highlights_found: int = 0
    clips_extracted: int = 0
    clips_total: int = 0
    current_step: str = ""
    error: str | None = None


# --- Caption & Render Settings ---


class CaptionStyle(BaseModel):
    """Customizable caption styling for the editor."""

    enabled: bool = False
    font_family: str = "Helvetica"
    font_size_ratio: float = 0.055
    text_color: str = "#ffffff"
    highlight_color: str = "#00e676"
    shadow_color: str = "#000000"
    position: float = 0.72  # vertical position (0=top, 1=bottom)
    style_preset: str = "opus"  # opus, karaoke, subtitle, minimal
    censor_profanity: bool = False
    diarize: bool = False
    speaker_colors: dict[str, str] = {}
    disabled_speakers: list[str] = []
    speaker_labels: dict[str, str] = {}  # speaker_id -> custom display name


class WatermarkSettings(BaseModel):
    """Image/logo watermark overlaid on the clip."""

    enabled: bool = False
    image_path: str | None = None
    position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right
    opacity: float = 0.85  # 0.0 - 1.0
    size_ratio: float = 0.12  # fraction of video width
    margin: int = 24  # pixels from edge


class ClipRenderSettings(BaseModel):
    """Per-clip render settings saved in the manifest."""

    aspect_ratio: str = "16:9"
    caption_style: CaptionStyle = CaptionStyle()
    intro_path: str | None = None
    outro_path: str | None = None
    watermark: WatermarkSettings = WatermarkSettings()


class RenderRequest(BaseModel):
    """Request to render a clip with specific settings."""

    aspect_ratio: str = "16:9"
    caption_style: CaptionStyle = CaptionStyle()
    intro_path: str | None = None
    outro_path: str | None = None
    watermark: WatermarkSettings = WatermarkSettings()
    openai_api_key: str | None = None  # Override for transcription
    hf_token: str | None = None  # Override for speaker diarization


class RenderStatus(str, Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class RenderResponse(BaseModel):
    """Response for a render job."""

    status: RenderStatus
    current_step: str | None = None
    output_path: str | None = None
    error: str | None = None


class SpeakerInfo(BaseModel):
    """Information about a detected speaker."""

    speaker_id: str  # e.g., SPEAKER_00
    sample_words: list[str]  # first few words spoken by this speaker
    word_count: int  # total words spoken
    total_duration_seconds: float


class TranscriptResult(BaseModel):
    """Cached transcript with speaker info."""

    words: list[dict]  # serialized TimedWord
    speakers: list[SpeakerInfo]
    diarized: bool


# --- Job Models (for API) ---


class JobRequest(BaseModel):
    """Request to process a video (API schema)."""

    input_path: str
    output_dir: str | None = None
    criteria: str | None = None
    preset: str | None = None
    gemini_model: str | None = None
    target_clips_per_hour: float = 2.5
    min_clip_seconds: float = 15.0
    margin_seconds: float = 3.0
    analysis_mode: str = "fast"
    accurate_cuts: bool = False
    captions: bool = False
    caption_model_size: str = "base"
    api_key: str | None = None
    openai_api_key: str | None = None


class JobResponse(BaseModel):
    """Response for a processing job."""

    job_id: str
    status: JobStatus
    progress: ProgressEvent | None = None
    highlights: list[DeduplicatedHighlight] | None = None
    clips: list[ExtractedClip] | None = None
    error: str | None = None


class ClipManifest(BaseModel):
    """Written to output_dir/manifest.json."""

    input_path: str
    video_duration_seconds: float
    settings_used: dict
    highlights: list[DeduplicatedHighlight]
    clips: list[ExtractedClip] = []
    clip_settings: dict[int, ClipRenderSettings] = {}
    session_defaults: ClipRenderSettings = ClipRenderSettings()
