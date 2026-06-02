"""Configuration: settings, enums, and defaults for OpenReel."""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class AspectRatio(str, Enum):
    """Output aspect ratio for clips."""

    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    SQUARE = "1:1"


class GeminiModel(str, Enum):
    """Supported Gemini models."""

    FLASH = "gemini-2.5-flash"
    PRO = "gemini-2.5-pro"
    FLASH_LITE = "gemini-2.0-flash-lite"


class AnalysisMode(str, Enum):
    """Analysis mode controlling chunk size and token density."""

    FAST = "fast"
    DETAILED = "detailed"


class PresetCriteria(str, Enum):
    """Preset highlight detection criteria."""

    GENERAL = "general"
    GAMING = "gaming"
    IRL = "irl"
    MUSIC = "music"
    SPORTS = "sports"
    EDUCATIONAL = "educational"


PRESET_CRITERIA_TEXT: dict[PresetCriteria, str] = {
    PresetCriteria.GENERAL: (
        "Look for the most compelling and highlight-worthy moments, including: "
        "dramatic or emotional moments, funny or entertaining segments, impressive "
        "displays of skill, surprising or unexpected events, memorable interactions, "
        "and peak energy or hype moments."
    ),
    PresetCriteria.GAMING: (
        "Look for the best gaming highlights, including: impressive plays and clutch "
        "moments, exciting kills or victories, funny fails or rage moments, close calls "
        "and near-death escapes, boss defeats or major achievements, and moments of high "
        "emotional reaction from the streamer."
    ),
    PresetCriteria.IRL: (
        "Look for the most engaging IRL stream moments, including: funny interactions "
        "with people, unexpected events or surprises, emotional or heartwarming moments, "
        "awkward or cringe-worthy situations, exciting activities or stunts, and memorable "
        "conversations or stories."
    ),
    PresetCriteria.MUSIC: (
        "Look for the best musical moments, including: impressive vocal or instrumental "
        "performances, crowd reactions and sing-alongs, emotional or powerful deliveries, "
        "funny or unexpected musical moments, genre transitions or mashups, and requests "
        "or dedications with audience interaction."
    ),
    PresetCriteria.SPORTS: (
        "Look for the most exciting sports moments, including: goals, scores, and key plays, "
        "dramatic comebacks and clutch performances, controversial calls or reactions, "
        "celebrations and emotional moments, injuries or unexpected stoppages, and "
        "commentator reactions to big moments."
    ),
    PresetCriteria.EDUCATIONAL: (
        "Look for the most valuable educational moments, including: key insights and 'aha' "
        "moments, clear explanations of complex topics, practical demonstrations and examples, "
        "important Q&A exchanges, surprising facts or revelations, and summary or recap segments."
    ),
}

# Token costs per second of video at each analysis mode
# Gemini's actual tokenization varies heavily by content complexity.
# Using 300 tokens/sec to ensure even dense video segments stay under
# the 1M token limit. Results in ~30 min chunks for 720p 2fps.
_TOKENS_PER_SECOND = {
    AnalysisMode.FAST: 300.0,
    AnalysisMode.DETAILED: 600.0,
}

# Reserve tokens for prompt + response (generous buffer)
_RESERVED_TOKENS = 150_000
_CONTEXT_WINDOW = 1_000_000


class OpenReelSettings(BaseSettings):
    """All configurable settings for OpenReel.

    Values are resolved in order: CLI flags > environment variables > defaults.
    Environment variables use the OPENREEL_ prefix (e.g., OPENREEL_GEMINI_MODEL).
    """

    model_config = {"env_prefix": "OPENREEL_", "env_file": ".env", "env_file_encoding": "utf-8"}

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key. Also reads from GEMINI_API_KEY env var.",
    )
    gemini_model: GeminiModel = Field(
        default=GeminiModel.FLASH,
        description="Gemini model to use for analysis.",
    )
    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.FAST,
        description="Analysis mode: 'fast' uses fewer chunks (cheaper), 'detailed' uses more (finer analysis).",
    )
    target_clips_per_hour: float = Field(
        default=2.5,
        ge=0.1,
        description="Target number of highlight clips to find per hour of video.",
    )
    min_clips_total: int = Field(
        default=6,
        ge=1,
        description="Minimum total highlights to find regardless of video duration.",
    )
    min_clip_seconds: float = Field(
        default=15.0,
        ge=1.0,
        description="Minimum clip duration in seconds.",
    )
    margin_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Padding added before and after each clip in seconds.",
    )
    overlap_seconds: float = Field(
        default=120.0,
        ge=0.0,
        description="Overlap between analysis chunks in seconds.",
    )
    downscale_for_analysis: bool = Field(
        default=False,
        description="Downscale video chunks to 720p before uploading to Gemini. Faster uploads but CPU-intensive.",
    )
    analysis_fps: int = Field(
        default=2,
        ge=1,
        le=60,
        description="FPS for analysis chunks uploaded to Gemini. Lower = smaller files, faster uploads. Gemini samples at ~1fps so 2 is plenty.",
    )
    max_analysis_concurrency: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent Gemini API requests.",
    )
    max_extraction_concurrency: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent ffmpeg extraction processes.",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="Output aspect ratio for clips: 16:9, 9:16, or 1:1.",
    )
    accurate_cuts: bool = Field(
        default=False,
        description="Re-encode clips for frame-accurate cuts (slower).",
    )
    captions: bool = Field(
        default=False,
        description="Add Opus-style auto-captions to clips.",
    )
    caption_model_size: str = Field(
        default="base",
        description="Whisper model size for captions: base (141 MB), small (466 MB), medium (1.5 GB).",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for Whisper transcription. If set, uses cloud API instead of local model.",
    )
    diarize: bool = Field(
        default=False,
        description="Enable speaker diarization for multi-speaker captions with per-speaker colors.",
    )
    hf_token: str = Field(
        default="",
        description="HuggingFace token for pyannote speaker diarization model access.",
    )

    @property
    def tokens_per_second(self) -> float:
        return _TOKENS_PER_SECOND[self.analysis_mode]

    @property
    def max_chunk_seconds(self) -> float:
        usable_tokens = _CONTEXT_WINDOW - _RESERVED_TOKENS
        return usable_tokens / self.tokens_per_second

    def resolve_api_key(self) -> str:
        """Return the API key, falling back to GEMINI_API_KEY env var or .env file."""
        import os

        from dotenv import load_dotenv

        load_dotenv()

        key = self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY or OPENREEL_GEMINI_API_KEY "
                "environment variable, or pass --api-key."
            )
        return key
