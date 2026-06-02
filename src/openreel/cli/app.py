"""OpenReel CLI — Typer application."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from openreel.config import (
    AnalysisMode,
    AspectRatio,
    GeminiModel,
    OpenReelSettings,
    PresetCriteria,
)
from openreel.models import DeduplicatedHighlight, JobStatus

app = typer.Typer(
    name="openreel",
    help="Automatically extract highlight clips from stream recordings using Gemini.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)

# Common options
InputVideo = Annotated[Path, typer.Argument(help="Path to the MP4 file to process.", exists=True)]
OutputDir = Annotated[
    Optional[Path],
    typer.Option("-o", "--output-dir", help="Output directory for clips."),
]
Criteria = Annotated[
    Optional[str],
    typer.Option("-c", "--criteria", help="Custom highlight criteria (free text)."),
]
Preset = Annotated[
    Optional[PresetCriteria],
    typer.Option("-p", "--preset", help="Preset criteria category."),
]
Model = Annotated[
    GeminiModel,
    typer.Option("-m", "--model", help="Gemini model to use."),
]
ClipsPerHour = Annotated[
    float,
    typer.Option("--clips-per-hour", help="Target highlights per hour."),
]
MinClipLength = Annotated[
    float,
    typer.Option("--min-clip-length", help="Minimum clip duration in seconds."),
]
Margin = Annotated[
    float,
    typer.Option("--margin", help="Padding before/after each clip in seconds."),
]
AnalysisModeOpt = Annotated[
    AnalysisMode,
    typer.Option("--analysis-mode", help="Analysis mode: 'fast' (fewer chunks) or 'detailed' (more chunks)."),
]
AspectRatioOpt = Annotated[
    AspectRatio,
    typer.Option("--aspect-ratio", help="Output aspect ratio: 16:9, 9:16, or 1:1."),
]
AccurateCuts = Annotated[
    bool,
    typer.Option("--accurate-cuts", help="Re-encode for frame-accurate cuts (slower)."),
]
NoDownscale = Annotated[
    bool,
    typer.Option("--no-downscale", help="Upload original resolution to Gemini (slower uploads)."),
]
AnalysisFps = Annotated[
    int,
    typer.Option("--analysis-fps", help="FPS for analysis chunks. Lower = smaller files. 0 = keep original."),
]
MinClipsTotal = Annotated[
    int,
    typer.Option("--min-clips", help="Minimum total highlights to find."),
]
Captions = Annotated[
    bool,
    typer.Option("--captions", help="Add Opus-style auto-captions to clips."),
]
CaptionModel = Annotated[
    str,
    typer.Option("--caption-model", help="Whisper model size: base, small, medium."),
]
Diarize = Annotated[
    bool,
    typer.Option("--diarize", help="Enable speaker diarization (color per speaker). Requires pyannote.audio."),
]
ApiKey = Annotated[
    Optional[str],
    typer.Option("--api-key", help="Gemini API key (overrides env var).", envvar="GEMINI_API_KEY"),
]
Verbose = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Enable verbose logging."),
]


def _build_settings(
    model: GeminiModel = GeminiModel.FLASH,
    clips_per_hour: float = 2.5,
    min_clips_total: int = 6,
    min_clip_length: float = 15.0,
    margin: float = 3.0,
    analysis_mode: AnalysisMode = AnalysisMode.FAST,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    accurate_cuts: bool = False,
    no_downscale: bool = False,
    analysis_fps: int = 2,
    captions: bool = False,
    caption_model: str = "base",
    diarize: bool = False,
    api_key: str | None = None,
) -> OpenReelSettings:
    """Build settings from CLI options."""
    overrides = {
        "gemini_model": model,
        "target_clips_per_hour": clips_per_hour,
        "min_clips_total": min_clips_total,
        "min_clip_seconds": min_clip_length,
        "margin_seconds": margin,
        "analysis_mode": analysis_mode,
        "aspect_ratio": aspect_ratio,
        "accurate_cuts": accurate_cuts,
        "downscale_for_analysis": not no_downscale,
        "analysis_fps": analysis_fps,
        "captions": captions,
        "caption_model_size": caption_model,
        "diarize": diarize,
    }
    if api_key:
        overrides["gemini_api_key"] = api_key

    return OpenReelSettings(**overrides)


def _setup_logging(verbose: bool) -> None:
    """Configure logging with rich handler."""
    import logging

    from rich.logging import RichHandler

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=False)],
    )
    # Suppress noisy HTTP logs from the Gemini SDK's transport layer
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.command()
def run(
    input_video: InputVideo,
    output_dir: OutputDir = None,
    criteria: Criteria = None,
    preset: Preset = None,
    model: Model = GeminiModel.FLASH,
    clips_per_hour: ClipsPerHour = 2.5,
    min_clips_total: MinClipsTotal = 6,
    min_clip_length: MinClipLength = 15.0,
    margin: Margin = 3.0,
    analysis_mode: AnalysisModeOpt = AnalysisMode.FAST,
    aspect_ratio: AspectRatioOpt = AspectRatio.LANDSCAPE,
    accurate_cuts: AccurateCuts = False,
    no_downscale: NoDownscale = False,
    analysis_fps: AnalysisFps = 2,
    captions: Captions = False,
    caption_model: CaptionModel = "base",
    diarize: Diarize = False,
    api_key: ApiKey = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Analyze only, don't extract clips.")] = False,
    verbose: Verbose = False,
) -> None:
    """Analyze a video and extract highlight clips."""
    _setup_logging(verbose)
    settings = _build_settings(
        model,
        clips_per_hour,
        min_clips_total,
        min_clip_length,
        margin,
        analysis_mode,
        aspect_ratio,
        accurate_cuts,
        no_downscale,
        analysis_fps,
        captions,
        caption_model,
        diarize,
        api_key,
    )

    try:
        settings.resolve_api_key()
    except ValueError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)

    from openreel.core.pipeline import process_video

    async def _run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=err_console,
        ) as progress:
            task = progress.add_task("Starting...", total=None)

            async for event in process_video(
                input_path=input_video,
                settings=settings,
                criteria=criteria,
                preset=preset,
                output_dir=output_dir,
                dry_run=dry_run,
            ):
                if event.status == JobStatus.ANALYZING:
                    if event.chunks_total > 0:
                        progress.update(
                            task,
                            description=event.current_step,
                            total=event.chunks_total,
                            completed=event.chunks_completed,
                        )
                    else:
                        progress.update(task, description=event.current_step)

                elif event.status == JobStatus.EXTRACTING:
                    progress.update(
                        task,
                        description=event.current_step,
                        total=event.clips_total,
                        completed=event.clips_extracted,
                    )

                elif event.status == JobStatus.COMPLETED:
                    progress.update(task, description="Complete", total=1, completed=1)
                    console.print(f"\n[green]{event.current_step}[/green]")

                elif event.status == JobStatus.FAILED:
                    progress.update(task, description=f"[red]{event.error}[/red]")
                    err_console.print(f"\n[red]Error:[/red] {event.error}")
                    raise typer.Exit(2)

    asyncio.run(_run())


@app.command()
def analyze(
    input_video: InputVideo,
    criteria: Criteria = None,
    preset: Preset = None,
    model: Model = GeminiModel.FLASH,
    clips_per_hour: ClipsPerHour = 2.5,
    min_clip_length: MinClipLength = 15.0,
    margin: Margin = 3.0,
    analysis_mode: AnalysisModeOpt = AnalysisMode.FAST,
    api_key: ApiKey = None,
    verbose: Verbose = False,
    output_format: Annotated[str, typer.Option("--format", help="Output format: json or table.")] = "json",
) -> None:
    """Analyze a video and output highlights as JSON (no clip extraction)."""
    _setup_logging(verbose)
    settings = _build_settings(model, clips_per_hour, min_clip_length, margin, analysis_mode, False, api_key)

    try:
        settings.resolve_api_key()
    except ValueError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(2)

    from openreel.core.pipeline import process_video

    final_event = None

    async def _run():
        nonlocal final_event
        async for event in process_video(
            input_path=input_video,
            settings=settings,
            criteria=criteria,
            preset=preset,
            dry_run=True,
        ):
            final_event = event
            if event.status == JobStatus.FAILED:
                err_console.print(f"[red]Error:[/red] {event.error}")
                raise typer.Exit(2)

    asyncio.run(_run())

    # Read the manifest and output highlights
    manifest_dir = Path("openreel_output") / input_video.stem
    manifest_path = manifest_dir / "manifest.json"

    if not manifest_path.exists():
        err_console.print("[red]Error:[/red] No manifest found after analysis.")
        raise typer.Exit(2)

    manifest_data = json.loads(manifest_path.read_text())
    highlights = manifest_data.get("highlights", [])

    if output_format == "json":
        console.print_json(json.dumps(highlights, indent=2))
    else:
        from rich.table import Table

        table = Table(title=f"Highlights — {input_video.name}")
        table.add_column("#", style="dim")
        table.add_column("Time")
        table.add_column("Title")
        table.add_column("Confidence")
        table.add_column("Tags")

        for h in highlights:
            start = h["start_seconds"]
            end = h["end_seconds"]
            time_str = f"{_fmt_time(start)} - {_fmt_time(end)}"
            table.add_row(
                str(h["index"] + 1),
                time_str,
                h["title"],
                f"{h['confidence']:.0%}",
                ", ".join(h["tags"]),
            )

        console.print(table)


@app.command()
def extract(
    input_video: InputVideo,
    highlights_file: Annotated[
        Path,
        typer.Option("--highlights", help="Path to JSON file with highlights (from analyze or manifest)."),
    ],
    output_dir: OutputDir = None,
    aspect_ratio: AspectRatioOpt = AspectRatio.LANDSCAPE,
    margin: Margin = 3.0,
    accurate_cuts: AccurateCuts = False,
    captions: Captions = False,
    caption_model: CaptionModel = "base",
    diarize: Diarize = False,
    verbose: Verbose = False,
) -> None:
    """Extract clips from a previously-generated highlights file."""
    _setup_logging(verbose)

    if not highlights_file.exists():
        err_console.print(f"[red]Error:[/red] Highlights file not found: {highlights_file}")
        raise typer.Exit(2)

    data = json.loads(highlights_file.read_text())

    # Accept either a raw list of highlights or a manifest with a "highlights" key
    if isinstance(data, dict) and "highlights" in data:
        highlights_raw = data["highlights"]
    elif isinstance(data, list):
        highlights_raw = data
    else:
        err_console.print("[red]Error:[/red] Invalid highlights file format.")
        raise typer.Exit(2)

    highlights = [DeduplicatedHighlight(**h) for h in highlights_raw]

    if not highlights:
        err_console.print("[yellow]No highlights to extract.[/yellow]")
        raise typer.Exit(0)

    if output_dir is None:
        output_dir = Path("openreel_output") / input_video.stem

    settings = OpenReelSettings(
        margin_seconds=margin,
        accurate_cuts=accurate_cuts,
        aspect_ratio=aspect_ratio,
        captions=captions,
        caption_model_size=caption_model,
        diarize=diarize,
    )

    from openreel.core.extractor import extract_all_clips

    async def _run():
        clips, errors = await extract_all_clips(
            input_path=input_video,
            highlights=highlights,
            output_dir=output_dir,
            settings=settings,
        )
        console.print(f"[green]Extracted {len(clips)} clips to {output_dir}[/green]")
        if errors:
            for err in errors:
                err_console.print(f"[yellow]Warning:[/yellow] {err}")

    asyncio.run(_run())


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload.")] = False,
    verbose: Verbose = False,
) -> None:
    """Start the OpenReel REST API server."""
    _setup_logging(verbose)
    import uvicorn

    console.print(f"Starting OpenReel API server on {host}:{port}...")
    uvicorn.run(
        "openreel.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="debug" if verbose else "info",
    )


def _fmt_time(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"
