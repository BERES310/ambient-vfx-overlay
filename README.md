# Ambient VFX Overlay

Turn a still photo into a living scene. The engine estimates depth, surface
normals, lighting and scene semantics (sky, ground, water, vegetation…) from a
single image and then renders GPU-accelerated weather on top of it: rain,
snow, fog, lightning, god rays, wind-swayed vegetation, water ripples, lens
effects and a matching colour grade.

Output can be a single PNG, a GIF, an MP4, or a **seamlessly looping MP4 live
wallpaper** (for Wallpaper Engine and similar players).

## Features

- **Depth-aware particles** – rain and snow pass *behind* foreground objects
  instead of floating flatly over the photo.
- **AI scene understanding** – Marigold depth / normals / intrinsic lighting,
  SegFormer segmentation (optional SAM refinement).
- **Physically-motivated weather** – motion-blurred rain streaks, soft snow,
  volumetric fog, lightning with scene flash, wet-surface and puddle effects.
- **Weather grades** – colour grading tuned per preset (`sunny`, `storm`, `snow`).
- **Seamless loops** – tail is crossfaded into the head; two-pass encode
  (lossless intermediate → CRF) so the GPU render is never repeated.
- **Friendly GUI** (`vfx_gui.py`) – pick a photo, pick a mood, watch a live
  preview, press *Make final video*. A **My videos** tab reloads the exact
  settings behind any finished video; plain-language error messages explain
  what went wrong (missing models, out of GPU memory, …).
- **Resume anywhere** – stop a render at any time and continue later, even
  after a crash or power cut.
- **Batch CLI** (`vfx_cli.py`) – many photos × many moods in one command.
- A single, heavily commented `config.py` for every low-level setting.

For a non-technical walkthrough see [`QUICKSTART.md`](QUICKSTART.md).

## Moods and weather presets

The GUI and CLI expose five moods, each with *intensity*, *wind* and
*darkness* sliders (0–1):

| Mood    | Contents                                                |
|---------|---------------------------------------------------------|
| `rain`  | rain with splashes, wet surfaces and light fog          |
| `snow`  | drifting snow with depth blur and snow grade            |
| `sunny` | sunbeams (god rays) and ambient light, no precipitation |
| `storm` | heavy rain, lightning, strong wind, storm grade         |
| `mist`  | low ground-hugging fog, soft dawn light                 |

Under the hood the batch tools also use the engine presets `rain_fog`, `snow`,
`clear_sunny` and `storm` (see `run_weather_benchmark.py`).

## Requirements

- Python 3.10+
- A GPU/driver with OpenGL 4.3+ (used through [moderngl](https://moderngl.readthedocs.io/))
- ~20 GB free disk for the AI models, a CUDA GPU strongly recommended for the
  diffusion models (CPU works but is slow)

## Installation

```bash
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/macOS:  source venv/bin/activate

# PyTorch first – pick the build for your machine (https://pytorch.org/get-started/locally/)
# NVIDIA GPU:
pip install torch torchvision
# CPU only (smaller):
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

pip install -r requirements.txt
```

## Downloading the models

The models are **not** included in this repository. The engine runs fully
offline (`HF_HUB_OFFLINE=1`) and looks for them in `models/huggingface/`, so
download them once before the first render:

```bash
python download_models.py          # required models (~15 GB)
python download_models.py --all    # also the optional metric-depth + SAM models
```

That script downloads these HuggingFace repositories into `models/huggingface/hub`:

| Repository | Used for | Required |
|------------|----------|----------|
| [`prs-eth/marigold-depth-v1-1`](https://huggingface.co/prs-eth/marigold-depth-v1-1) | depth (default) | yes |
| [`prs-eth/marigold-normals-v1-1`](https://huggingface.co/prs-eth/marigold-normals-v1-1) | surface normals | yes |
| [`prs-eth/marigold-iid-lighting-v1-1`](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) | learned illumination | yes |
| [`nvidia/segformer-b5-finetuned-ade-640-640`](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | scene segmentation | yes |
| [`depth-anything/Depth-Anything-V2-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | fast relative depth (`DEPTH_MODE = "relative"`) | yes |
| [`depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf) / [`…Indoor…`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf) | metric depth (`DEPTH_MODE = "metric"`) | `--all` |
| [`facebook/sam-vit-huge`](https://huggingface.co/facebook/sam-vit-huge) | SAM mask refinement (`ENABLE_SAM_SEGMENTATION`) | `--all` |

Manual alternative (same result):

```bash
pip install -U huggingface_hub
huggingface-cli download prs-eth/marigold-depth-v1-1 --cache-dir models/huggingface/hub
# …repeat for each repository above
```

Check each model's page for its licence before redistributing anything derived from it.

## Usage

Put your photos (`.png`, `.jpg`) in `input_images/`, then:

```bash
python vfx_gui.py                 # desktop GUI with live preview (Windows: double-click Start.bat)
python vfx_cli.py photo.jpg --mood storm --quality draft --length 10
python vfx_cli.py --all-photos --mood rain snow    # batch, one summary at the end
python main.py                    # one output per photo, format set by config.OUTPUT_MODE
python live_wallpaper.py          # seamless looping MP4 -> output/live_wallpapers/
python run_weather_benchmark.py   # every photo x every weather preset, with timing CSV
```

`vfx_cli.py` options: `--mood` (rain, snow, sunny, storm, mist), `--quality`
(draft, balanced, high, maximum), `--length`, `--fps`, `--intensity`, `--wind`,
`--darkness`, `--list-moods`, `--dry-run`. Defaults come from `settings.json`.

Results are written to `output/` (finished videos in `output/live_wallpapers/`).
On Windows, `Start.bat` creates a private virtual environment on first run and
launches the GUI.

## Configuration

Everything lives in [`config.py`](config.py), heavily commented. Common knobs:

| Setting | Meaning |
|---------|---------|
| `OUTPUT_MODE` | `"png"`, `"gif"` or `"mp4"` |
| `VIDEO_DURATION_SECONDS`, `FPS` | length / frame rate of animated output |
| `ENABLE_*` flags | switch rain, snow, fog, lightning, god rays… on or off |
| `DEPTH_MODE` | `"marigold"` (best), `"relative"` (fast), `"metric"` |
| `WEATHER_GRADE_STYLE` | `None`, `"sunny"`, `"storm"`, `"snow"` |
| `LIVE_WALLPAPER_*` | loop length and crossfade for `live_wallpaper.py` |

User-facing defaults (mood, quality, length, fps, intensity, wind, darkness)
are kept separately in `settings.json`.

The first run on a photo is slower because depth/normals/segmentation are
computed and cached (`cache/`); later runs on the same photo reuse them.

## Project layout

```
main.py                  batch renderer (png / gif / mp4)
live_wallpaper.py        seamless looping wallpaper renderer
vfx_gui.py               CustomTkinter GUI
vfx_cli.py               command-line batch renderer
Start.bat                Windows one-click launcher
QUICKSTART.md            plain-language user guide
settings.json            user-facing defaults
fast_preview.py          cheap single-frame preview path
run_weather_benchmark.py photos x weather presets batch
download_models.py       fetches the AI models
config.py                all settings
vfx_engine/              engine modules + GLSL shaders
  gui_*.py               GUI moods, system helpers and widgets
  error_hints.py         plain-language error explanations
  user_settings.py       settings.json handling
```

## Tips

- If a render looks too strong/weak, adjust intensity in `vfx_engine/particles_rain.py`
  (`_INTENSITY_PRESETS`) or the matching `config.py` entries.
- Windows: run from an activated venv so `python` resolves to the right interpreter.
