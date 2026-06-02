"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache

from openreel.config import OpenReelSettings


@lru_cache
def get_settings() -> OpenReelSettings:
    """Return the global settings instance."""
    return OpenReelSettings()
