"""
gui_system.py

Non-visual helpers behind vfx_gui.py. Nothing here
imports tkinter, so it can be tested headlessly.

Everything stays inside vfx_overlay/: models are downloaded into
config.MODELS_DIR/huggingface/hub, timing history and the pending-render
marker live in gui_presets/.
"""

import json
import os
import shutil
import statistics
import threading
from datetime import datetime
from pathlib import Path

import config

GUI_PRESETS_DIR = Path(__file__).resolve().parent.parent / "gui_presets"
TIMING_PATH = GUI_PRESETS_DIR / "timing.json"
PENDING_PATH = GUI_PRESETS_DIR / "pending_render.json"

# Serialises every use of the module-level `config.*` overrides: the live draft
# worker and the final-render worker both hold it while their overrides are applied.
CONFIG_LOCK = threading.RLock()

# Same list share/download_models.py treats as required. (repo id, what it is for)
REQUIRED_MODELS = [
    ("prs-eth/marigold-depth-v1-1", "depth of the scene"),
    ("prs-eth/marigold-normals-v1-1", "surface directions"),
    ("prs-eth/marigold-iid-lighting-v1-1", "lighting"),
    ("nvidia/segformer-b5-finetuned-ade-640-640", "sky, ground and plants"),
    ("depth-anything/Depth-Anything-V2-Large-hf", "depth (second opinion)"),
]


# ---------------------------------------------------------------------------
# First-run checks
# ---------------------------------------------------------------------------
def models_hub_dir():
    return config.MODELS_DIR / "huggingface" / "hub"


def model_folder(repo_id):
    return models_hub_dir() / ("models--" + repo_id.replace("/", "--"))


def model_is_present(repo_id):
    """A model counts as present once its snapshot folder holds at least one file."""
    snapshots = model_folder(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(child.is_dir() and any(child.iterdir()) for child in snapshots.iterdir())


def missing_models():
    return [(repo, purpose) for repo, purpose in REQUIRED_MODELS if not model_is_present(repo)]


def gpu_name():
    """Name of the CUDA graphics card, or None when there is none (or torch is missing)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - a broken torch install just means "no usable GPU"
        pass
    return None


def ffmpeg_ok():
    try:
        from vfx_engine.image_io import get_ffmpeg_path
        return Path(get_ffmpeg_path()).exists()
    except Exception:  # noqa: BLE001
        return False


def check_system():
    """{'gpu': name or None, 'ffmpeg': bool, 'missing_models': [(repo, purpose)]}"""
    return {"gpu": gpu_name(), "ffmpeg": ffmpeg_ok(), "missing_models": missing_models()}


def download_models(repos, on_progress=None):
    """
    Downloads the given model repos into config.MODELS_DIR (never a global
    cache). on_progress(repo_id, index, total) is called before each one.
    The engine runs with HF_HUB_OFFLINE=1; that is lifted only for the
    duration of the download and put back afterwards. Returns the repos that
    failed as [(repo, message)].
    """
    from huggingface_hub import constants, snapshot_download

    cache_dir = models_hub_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved_env = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    saved_flag = getattr(constants, "HF_HUB_OFFLINE", None)
    failures = []
    try:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        constants.HF_HUB_OFFLINE = False
        for index, repo in enumerate(repos):
            if on_progress is not None:
                on_progress(repo, index, len(repos))
            try:
                snapshot_download(repo_id=repo, cache_dir=str(cache_dir))
            except Exception as exc:  # noqa: BLE001 - report per model, keep going
                failures.append((repo, str(exc)))
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if saved_flag is not None:
            constants.HF_HUB_OFFLINE = saved_flag
    return failures


# ---------------------------------------------------------------------------
# Time estimates: seconds per megapixel-frame, learned from earlier renders
# ---------------------------------------------------------------------------
def _load_timing():
    try:
        with open(TIMING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [float(v) for v in data.get("seconds_per_megapixel_frame", []) if float(v) > 0]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def record_render_timing(total_seconds, width, height, frames):
    """Stores how long one finished render took, normalised per megapixel-frame."""
    if frames <= 0 or total_seconds <= 0:
        return
    value = total_seconds / (width * height / 1e6 * frames)
    history = (_load_timing() + [value])[-8:]
    try:
        GUI_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(TIMING_PATH, "w", encoding="utf-8") as f:
            json.dump({"seconds_per_megapixel_frame": history}, f, indent=2)
    except OSError:
        pass


def estimate_seconds(width, height, frames, draft_seconds_per_megapixel_frame=None, draft_fudge=3.0):
    """
    (seconds, basis) for a render of frames at width x height.
    basis is "history" (median of earlier renders on this machine), "draft"
    (extrapolated from the live draft - rough) or None when nothing is known.
    """
    history = _load_timing()
    megapixel_frames = width * height / 1e6 * frames
    if history:
        return statistics.median(history) * megapixel_frames, "history"
    if draft_seconds_per_megapixel_frame:
        return draft_seconds_per_megapixel_frame * draft_fudge * megapixel_frames, "draft"
    return None, None


# ---------------------------------------------------------------------------
# Interrupted-render marker (so the next launch can offer "Resume")
# ---------------------------------------------------------------------------
def write_pending(state):
    try:
        GUI_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        temp = PENDING_PATH.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(temp, PENDING_PATH)
    except OSError:
        pass


def read_pending():
    try:
        with open(PENDING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data.get("photo") else None
    except (OSError, ValueError):
        return None


def clear_pending():
    try:
        PENDING_PATH.unlink()
    except OSError:
        pass


def discard_render_checkpoints():
    """Throws away every saved partial render so the next one starts from frame 1."""
    directory = config.CACHE_DIR / "render_checkpoints"
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# "Set as wallpaper"
# ---------------------------------------------------------------------------
def extract_first_frame(video_path, out_path):
    """Saves the video's first frame as a PNG (via OpenCV). Returns out_path."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame from {video_path}")
    finally:
        capture.release()
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"Could not write {out_path}")
    return out_path


def set_desktop_picture(image_path):
    """Sets a still image as the Windows desktop picture (Windows only)."""
    import ctypes

    SPI_SETDESKWALLPAPER, SPIF_UPDATEINIFILE, SPIF_SENDCHANGE = 20, 1, 2
    ok = ctypes.windll.user32.SystemParametersInfoW(
        SPI_SETDESKWALLPAPER, 0, str(Path(image_path).resolve()), SPIF_UPDATEINIFILE | SPIF_SENDCHANGE)
    if not ok:
        raise RuntimeError("Windows did not accept the picture as a wallpaper.")


# ---------------------------------------------------------------------------
# My videos: everything in output/live_wallpapers with what made it
# ---------------------------------------------------------------------------
def list_videos(directory=None):
    """Newest first: [{'path', 'size_mb', 'modified', 'photo', 'config'}]."""
    directory = Path(directory or config.LIVE_WALLPAPERS_DIR)
    videos = []
    for path in directory.glob("*.mp4"):
        info = {"path": path, "size_mb": path.stat().st_size / 1e6,
                "modified": datetime.fromtimestamp(path.stat().st_mtime), "photo": None, "config": {}}
        sidecar = path.with_name(path.name + ".provenance.json")
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            info["photo"] = data.get("source_image")
            info["config"] = data.get("config", {})
        except (OSError, ValueError):
            pass
        videos.append(info)
    videos.sort(key=lambda v: v["modified"], reverse=True)
    return videos


# ---------------------------------------------------------------------------
# Wallpaper Engine export
# ---------------------------------------------------------------------------
def export_wallpaper_engine(video_path, title=None, out_root=None):
    """
    Packages a finished video as a Wallpaper Engine "video" project: a folder
    holding the mp4, a preview image and the project.json Wallpaper Engine reads.
    Lives in output/live_wallpapers/wallpaper_engine/<name>/ - copy that folder
    into Wallpaper Engine's projects/myprojects/ folder (or drag it into the
    editor). Returns the folder.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    root = Path(out_root or (config.LIVE_WALLPAPERS_DIR / "wallpaper_engine"))
    folder = root / video_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, folder / video_path.name)
    extract_first_frame(video_path, folder / "preview.png")
    project = {
        "description": "Made with VFX Overlay",
        "file": video_path.name,
        "preview": "preview.png",
        "title": title or video_path.stem.replace("_", " "),
        "type": "video",
        "tags": ["Nature"],
        "contentrating": "Everyone",
        "general": {},
        "version": 1,
    }
    with open(folder / "project.json", "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)
    return folder
