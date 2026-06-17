# OpenReel

Automatically identify and extract highlight clips from long stream recordings using Google Gemini's video understanding.

Feed OpenReel a 5+ hour stream VOD and it will:
1. Split the video into chunks that fit Gemini's context window
2. Analyze each chunk to find the most highlight-worthy moments
3. Deduplicate moments found near chunk boundaries
4. Extract individual clips with configurable margin/padding via ffmpeg
5. Optionally add Opus-style auto-captions with speaker diarization
6. Output clips in any aspect ratio (16:9, 9:16, 1:1)

---

## Quick Start

### One command (recommended)

```bash
bash start.sh
```

This handles everything — installs prerequisites if needed, creates a virtual environment, prompts for your API key, and starts the server. On repeat runs it skips straight to launching.

To set up without starting the server, use `bash setup.sh` instead.

### Manual setup

```bash
python3 -m venv .venv && source .venv/bin/activate  # see below for Windows
pip install -e .
cp .env.example .env   # then add your GEMINI_API_KEY
openreel run stream.mp4
```

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime |
| ffmpeg | Any recent | Video splitting & clip extraction |
| ffprobe | Ships with ffmpeg | Video metadata extraction |
| Gemini API key | — | Video analysis ([get one free](https://aistudio.google.com/apikey)) |

### Install Python

| Platform | Command |
|----------|---------|
| macOS | `brew install python@3.12` or download from [python.org](https://www.python.org/downloads/) |
| Ubuntu/Debian | `sudo apt update && sudo apt install python3.12 python3.12-venv` |
| Fedora | `sudo dnf install python3.12` |
| Arch | `sudo pacman -S python` |
| Windows | `winget install Python.Python.3.12` or download from [python.org](https://www.python.org/downloads/) |

Verify: `python3 --version` (or `python --version` on Windows)

### Install ffmpeg

| Platform | Command |
|----------|---------|
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| Windows | `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH |

Verify: `ffmpeg -version` and `ffprobe -version`

> **Windows PATH note:** If you downloaded ffmpeg manually, extract it and add the `bin/` folder to your system PATH. Search "Environment Variables" in Windows Settings to edit PATH.

---

## Installation

### macOS / Linux

```bash
git clone https://github.com/nfteague/open-reel.git
cd open-reel

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install OpenReel
pip install -e .
```

### Windows (PowerShell)

```powershell
git clone https://github.com/nfteague/open-reel.git
cd open-reel

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install OpenReel
pip install -e .
```

### Windows (Command Prompt)

```cmd
git clone https://github.com/nfteague/open-reel.git
cd open-reel

python -m venv .venv
.venv\Scripts\activate.bat

pip install -e .
```

> **Optional extras (opt-in).** The base install above keeps things lean. Auto-captions
> and speaker diarization are separate, heavier downloads you only add if you need them:
> `pip install -e ".[captions]"` (local transcription) — see [Auto-Captions](#auto-captions).

---

## Configuration

### API Key (required)

Get a free Gemini API key at https://aistudio.google.com/apikey

**Option A — `.env` file (recommended, works on all platforms):**

```bash
cp .env.example .env
```

Then edit `.env` and replace `your-api-key-here` with your actual key:

```
GEMINI_API_KEY=AIza...
```

**Option B — Environment variable:**

```bash
# macOS / Linux
export GEMINI_API_KEY=AIza...

# Windows PowerShell
$env:GEMINI_API_KEY="AIza..."

# Windows Command Prompt
set GEMINI_API_KEY=AIza...
```

### Optional Settings

All settings use the `OPENREEL_` prefix as environment variables or in your `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key (required) |
| `OPENREEL_GEMINI_MODEL` | `gemini-2.5-flash` | Model: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash-lite` |
| `OPENREEL_ANALYSIS_MODE` | `fast` | `fast` (~2.5h per chunk, cheaper) or `detailed` (~52min per chunk, finer) |
| `OPENREEL_TARGET_CLIPS_PER_HOUR` | `2.5` | Highlights to find per hour of video |
| `OPENREEL_MIN_CLIP_SECONDS` | `15` | Minimum clip duration in seconds |
| `OPENREEL_MARGIN_SECONDS` | `3.0` | Padding added before/after each clip |
| `OPENREEL_ACCURATE_CUTS` | `false` | Re-encode for frame-accurate cuts (slower) |
| `OPENREEL_ANALYSIS_FPS` | `2` | FPS for analysis uploads (lower = faster uploads) |
| `OPENREEL_DOWNSCALE_FOR_ANALYSIS` | `false` | Downscale to 720p before uploading |

---

## Verify it works

Before pointing OpenReel at a multi-hour VOD, confirm the install in under a minute:

```bash
# 1. CLI is on PATH (venv must be active)
openreel --help

# 2. Health check — ffmpeg, ffprobe, and your API key should all be green
openreel serve &
curl http://localhost:8000/health
# → {"healthy":true,"ffmpeg":"ok","ffprobe":"ok","gemini_api_key":"configured",...}

# 3. End-to-end dry run on a short clip (analysis only, no upload of a huge file)
#    Use any short .mp4 you have handy; --dry-run writes a manifest without extracting.
openreel run sample.mp4 --dry-run
```

If `/health` shows `"healthy": true` and the dry run produces a `manifest.json`, your
0→1 setup is complete. (`gemini_api_key` is read from your `.env` automatically — no need
to `export` it.)

---

## Usage

### Common Options

```bash
# Custom highlight criteria
openreel run stream.mp4 -c "Look for funny moments and big plays"

# Use a preset (gaming, irl, music, sports, educational)
openreel run stream.mp4 -p gaming

# More clips per hour, shorter minimum length
openreel run stream.mp4 --clips-per-hour 5 --min-clip-length 10

# Use Gemini Pro for better analysis (costs more)
openreel run stream.mp4 -m pro

# Dry run — analyze only, write manifest without extracting
openreel run stream.mp4 --dry-run

# Frame-accurate cuts (slower, re-encodes)
openreel run stream.mp4 --accurate-cuts

# Custom output directory
openreel run stream.mp4 -o ./my-clips

# Output as 9:16 vertical (TikTok/Reels/Shorts)
openreel run stream.mp4 --aspect-ratio 9:16

# Output as 1:1 square (Instagram)
openreel run stream.mp4 --aspect-ratio 1:1

# Add Opus-style auto-captions
openreel run stream.mp4 --captions

# Captions with speaker diarization (each speaker gets a unique color)
openreel run stream.mp4 --captions --diarize
```

### Analyze Only

Output highlights as JSON or a table (no clip extraction):

```bash
openreel analyze stream.mp4
openreel analyze stream.mp4 --format table
```

### Extract from Manifest

Run analysis first, curate the results, then extract only the clips you want:

```bash
openreel run stream.mp4 --dry-run
# Edit openreel_output/stream/manifest.json to remove unwanted highlights
openreel extract stream.mp4 --highlights openreel_output/stream/manifest.json
```

### Web Dashboard & API Server

OpenReel includes a web-based dashboard for browsing, editing, and rendering clips:

```bash
openreel serve
```

Open http://localhost:8000 in your browser. The dashboard lets you:
- Browse and select video files
- Submit processing jobs
- Preview detected highlights
- Customize aspect ratio, captions, watermarks, and intros/outros per clip
- Render final clips with your settings

Server options:

```bash
openreel serve --port 9000       # custom port
openreel serve --reload          # auto-reload on code changes (development)
openreel serve --host 127.0.0.1  # bind to localhost only
```

---

## Auto-Captions

OpenReel can burn Opus-style captions into clips — word-by-word display with the active word highlighted.

```bash
# Install the captions extra
pip install -e ".[captions]"

# Basic captions (uses local faster-whisper)
openreel run stream.mp4 --captions

# Use a larger Whisper model for better accuracy
openreel run stream.mp4 --captions --caption-model small
```

### Speaker Diarization

When multiple people are talking, `--diarize` assigns each speaker a unique caption color.

**Additional requirements:**

1. Install pyannote: `pip install pyannote.audio`
2. Accept model terms on HuggingFace:
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
3. Create an access token at [hf.co/settings/tokens](https://hf.co/settings/tokens) (Read permission)

```bash
# Set your HuggingFace token
export OPENREEL_HF_TOKEN=hf_...  # or add to .env

# Run with diarization
openreel run stream.mp4 --captions --diarize
```

---

## Preset Criteria

| Preset | Best For |
|--------|----------|
| `general` | Any content (default) |
| `gaming` | Gameplay — kills, clutches, reactions |
| `irl` | IRL streams — interactions, surprises, stories |
| `music` | Music — performances, reactions |
| `sports` | Sports — goals, comebacks |
| `educational` | Tutorials — key insights, demos, Q&A |

---

## Output Structure

```
openreel_output/stream/
├── clip_001_00h12m34s_insane_triple_kill.mp4
├── clip_002_01h05m12s_funny_donation_reaction.mp4
├── clip_003_02h30m45s_clutch_victory.mp4
├── rendered/                  # crops, captions, watermarked versions
│   └── clip_001_..._9x16.mp4
└── manifest.json              # full metadata, settings, timestamps
```

---

## REST API Reference

When running `openreel serve`, the full OpenAPI docs are at http://localhost:8000/docs.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jobs` | Submit a processing job |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{id}` | Poll job status and progress |
| `GET` | `/jobs/{id}/highlights` | Get detected highlights |
| `GET` | `/jobs/{id}/clips` | Get extracted clips |
| `DELETE` | `/jobs/{id}` | Cancel or remove a job |
| `POST` | `/jobs/{id}/render/{clip_index}` | Render clip with effects |
| `GET` | `/jobs/{id}/render/{clip_index}/status` | Poll render progress |
| `POST` | `/jobs/{id}/clips/{clip_index}/transcribe` | Transcribe clip audio |
| `GET` | `/jobs/{id}/clips/{clip_index}/transcribe/status` | Poll transcription status |
| `GET` | `/jobs/{id}/thumbnail/{clip_index}` | Get clip thumbnail |
| `GET` | `/jobs/{id}/session-defaults` | Get session render defaults |
| `PUT` | `/jobs/{id}/session-defaults` | Set session render defaults |
| `GET` | `/probe?path=...` | Probe video file metadata |
| `GET` | `/browse?path=...` | Browse filesystem for videos |
| `GET` | `/fonts` | List available caption fonts |
| `GET` | `/asset?path=...` | Serve local media file |
| `GET` | `/health` | System health check |

### Example API Usage

```bash
# Submit a job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/path/to/stream.mp4", "preset": "gaming"}'

# Poll status
curl http://localhost:8000/jobs/{job_id}

# Check system health
curl http://localhost:8000/health
```

---

## How It Works

1. **Probe** — Extract video metadata (duration, resolution, codec) via ffprobe
2. **Chunk** — Split the timeline into overlapping windows that fit Gemini's 1M token context
3. **Split** — Create physical chunk files via `ffmpeg -c copy` (fast, no re-encoding)
4. **Upload** — Upload each chunk to Gemini's File API
5. **Analyze** — Send each chunk with a structured prompt; Gemini returns JSON with timestamps, titles, descriptions, and confidence scores
6. **Deduplicate** — Merge highlights found in overlapping regions between chunks
7. **Extract** — Cut clips from the source video with configurable margin

---

## Setup for AI Agents (Claude Code, Cursor, Codex, etc.)

If a user asks you (an AI agent) to "install and run OpenReel," follow this flow.
**Steps 1–2 are silent; step 3 you must STOP and ASK the user before continuing.**

### 1. Clone and check prerequisites

```bash
git clone https://github.com/nfteague/open-reel.git && cd open-reel
python3 --version   # needs 3.11+
ffmpeg -version     # must be installed (ffprobe ships with it)
```

### 2. Create the virtual environment and install the base package

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

### 3. STOP and ask the user two questions

Do not assume answers — ask, then act on the replies:

1. **"Do you want auto-captions?"** Explain the tradeoff: captions burn word-by-word
   subtitles into clips, but require an extra, heavier download (`faster-whisper`).
   - If **yes** → `pip install -e ".[captions]"`
   - If **no** → skip it (they can add it later)
2. **"What's your Gemini API key?"** Tell them it's free at
   https://aistudio.google.com/apikey (the key starts with `AIza`). Then write it to `.env`:

```bash
cp .env.example .env
# Replace your-api-key-here with the key the user gave you:
# GEMINI_API_KEY=AIza...
```

### 4. Finish setup and start the server

```bash
curl -s http://localhost:8000/health   # before: confirm prerequisites; expect healthy:true after key is set
openreel serve                          # starts the dashboard at http://localhost:8000
```

### 5. Tell the user it's ready

Let them know the server is running at http://localhost:8000, **and that they can point
you at a video** to process it for them — e.g. they say "run it on ~/Downloads/stream.mp4"
and you run:

```bash
openreel run /path/to/video.mp4 -p gaming   # presets: general, gaming, irl, music, sports, educational
```

**Notes for agents:**
- The `.env` file in the project root is auto-loaded — no need to `export` variables (this applies to `/health` too)
- The `openreel` CLI is available after `pip install -e .` (registered as a console script)
- The web dashboard at http://localhost:8000 is a GUI alternative to the CLI
- All output goes to `./openreel_output/<video_stem>/` by default

---

## Troubleshooting

### "command not found: openreel"

Make sure your virtual environment is activated:

```bash
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\Activate.ps1   # Windows PowerShell
```

### "No Gemini API key found"

Ensure `GEMINI_API_KEY` is set. Check with:

```bash
# macOS/Linux
echo $GEMINI_API_KEY

# Windows PowerShell
echo $env:GEMINI_API_KEY
```

Or verify your `.env` file exists in the project root and contains `GEMINI_API_KEY=...`.

### "ffmpeg: command not found"

Install ffmpeg (see [Prerequisites](#install-ffmpeg)) and ensure it's on your PATH:

```bash
which ffmpeg    # macOS/Linux — should print a path
where ffmpeg    # Windows — should print a path
```

### Windows: "running scripts is disabled"

If PowerShell blocks the venv activation script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Captions not working

Install the captions extra:

```bash
pip install -e ".[captions]"
```

This installs `faster-whisper` for local transcription.

### Diarization errors

Speaker diarization requires:
1. `pip install pyannote.audio`
2. A HuggingFace token with access to the pyannote models (see [Speaker Diarization](#speaker-diarization))
3. `OPENREEL_HF_TOKEN` set in your `.env` or environment

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

---

## License

GPL-3.0-or-later

