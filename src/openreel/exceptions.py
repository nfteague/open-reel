"""Exception hierarchy for OpenReel."""


class OpenReelError(Exception):
    """Base exception for all OpenReel errors."""


class VideoFileError(OpenReelError):
    """Input video issues: not found, unreadable, too short, wrong format."""


class FFmpegError(OpenReelError):
    """ffmpeg or ffprobe execution failure."""


class GeminiError(OpenReelError):
    """Base for Gemini API-related errors."""


class FileUploadError(GeminiError):
    """File API upload or processing failure."""


class AnalysisError(GeminiError):
    """generate_content call failure or unparseable response."""


class RateLimitError(GeminiError):
    """429 rate limit from Gemini API."""


class ClipExtractionError(OpenReelError):
    """Individual clip extraction failure."""
