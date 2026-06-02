"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from openreel.api.routes import _jobs, load_persisted_jobs, router

logger = logging.getLogger(__name__)

_FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: reload persisted jobs
    load_persisted_jobs()
    logger.info("Loaded %d persisted job(s)", len(_jobs))
    yield
    # Shutdown: cancel running jobs, persist final state
    for job_id, job in list(_jobs.items()):
        if job.task and not job.task.done():
            logger.info("Cancelling job %s on shutdown", job_id)
            job.task.cancel()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="OpenReel",
        description="Automatically extract highlight clips from stream recordings using Gemini.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    # Serve extracted clips from any output directory
    @app.get("/clips/{path:path}")
    async def serve_clip(path: str):
        """Serve an extracted clip file."""
        clip_path = Path(path)
        if not clip_path.exists() or not clip_path.suffix == ".mp4":
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Clip not found")
        return FileResponse(clip_path, media_type="video/mp4")

    # Serve frontend
    @app.get("/")
    async def index():
        return FileResponse(_FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    return app
