"""
image_io.py

Loading input photos and saving the finished result - as a static image, an
animated GIF, or an MP4 video file (choice in config.py, the OUTPUT_MODE
setting).
"""

import os
import subprocess
import threading

import numpy as np
from PIL import Image

import config
from vfx_engine import color

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_input_images(folder):
    """Returns a sorted list of paths to every photo in the input folder."""
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_base_image(path):
    """
    Loads the input photo, converts it to RGB mode (no alpha channel), and,
    if it's larger than config.MAX_INPUT_DIMENSION, downscales it (keeping
    the aspect ratio) - see the comment next to that setting in config.py.
    """
    image = Image.open(path).convert("RGB")

    max_dim = config.MAX_INPUT_DIMENSION
    if max_dim is not None and max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def encode_frames_to_images(frames):
    """
    Generator - converts the linear-light float32 (height, width, 3)
    numpy arrays this pipeline's compositing/post-processing chain
    operates on
    back to gamma-encoded 8-bit PIL RGB Images, exactly ONCE, right here -
    the single point where the linear pipeline ends and the
    save_result/save_mp4/_save_gif functions below (which all expect a
    plain PIL Image, unchanged from before this pipeline existed) pick
    back up.
    """
    for frame in frames:
        yield color.encode_image(frame, mode="RGB")


def save_result(frames, input_path, output_mode, suffix=""):
    """
    Saves the finished frame sequence (any iterable - a list or a
    generator, frames are saved on the fly, without holding all of them in
    memory at once) in the output/ folder, in the format chosen in
    config.py (OUTPUT_MODE = "image" / "gif" / "mp4", plain extensions like
    "png" are also accepted - see config.resolve_output_mode).
    Returns the path of the saved file.

    suffix: appended to the filename (e.g. "_rain", "_snow" - see
        main.resolve_effect_suffix) so re-running with different effects
        enabled doesn't silently overwrite a previous, different-looking
        result under the same name.
    """
    stem = input_path.stem + suffix

    if output_mode == "image":
        out_path = config.OUTPUT_DIR / f"{stem}.png"
        next(iter(frames)).save(out_path)
    elif output_mode == "gif":
        out_path = config.OUTPUT_DIR / f"{stem}.gif"
        _save_gif(frames, out_path)
    elif output_mode == "mp4":
        out_path = config.OUTPUT_DIR / f"{stem}.mp4"
        save_mp4(frames, out_path)
    else:
        raise ValueError(f"Unknown OUTPUT_MODE: '{output_mode}'")

    return out_path


def _save_gif(frames, out_path):
    """
    Saves frames as an animated GIF (Pillow can do this with no extra
    libraries).

    Explicitly quantizes every frame to ONE shared adaptive palette (built
    once from the first frame - a reasonable proxy for the whole clip's
    color range, since every frame is the same photo with effects drifting
    over it) with Floyd-Steinberg dithering - GIF's 256-color
    palette is a hard format limitation this can't remove, but a shared
    palette avoids visible color-palette drift/flicker between frames that
    letting Pillow's save_all quantize each frame independently could
    introduce, and dithering breaks up the resulting banding on gradient-
    heavy content (sky, haze, bloom) into a fine, far-less-visible pattern
    instead of hard color-step banding.
    """
    duration_ms = int(1000 / config.FPS)
    frame_iter = iter(frames)
    first_frame = next(frame_iter)

    palette_source = first_frame.quantize(colors=256, method=Image.Quantize.MEDIANCUT)

    def quantized(frame):
        return frame.quantize(palette=palette_source)   # dither defaults to Dither.FLOYDSTEINBERG

    palette_source.save(
        out_path,
        save_all=True,
        append_images=(quantized(frame) for frame in frame_iter),   # generator - Pillow consumes it frame by frame
        duration=duration_ms,
        loop=0,
    )


def get_ffmpeg_path():
    """
    Returns the path to the ffmpeg binary from the `imageio-ffmpeg` package
    (see requirements.txt) - a ready-made, known-good .exe bundled with the
    pip package. We deliberately do NOT use whatever "ffmpeg" happens to be
    in the current terminal/IDE's PATH: different machines/terminals can
    have wildly different ffmpeg builds there (missing libx264, ancient
    versions, or on Windows even a Microsoft Store stub that does nothing),
    and that inconsistency is exactly what can make this work in one
    terminal and fail with "unsupported format" in another. Using the
    bundled binary makes the result identical no matter where this runs.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "Could not find a usable ffmpeg (the imageio-ffmpeg package is missing "
            "or broken) - it's needed to save video output. "
            "Install the dependencies from requirements.txt (pip install -r requirements.txt)."
        ) from error


def save_mp4(frames, out_path):
    """
    Saves frames as an .mp4 video file (H.264, single-pass) - used by the
    regular ambient-video pipeline (main.py). See save_mp4_two_pass() for
    the two-pass version used for live wallpapers.

    Encode settings (config.VIDEO_CRF/VIDEO_PRESET/VIDEO_PIXEL_FORMAT) are
    tuned for "quality is the only priority" - the previous
    version passed no -crf/-preset at all, silently falling back to
    ffmpeg's own "reasonable web video" build defaults (preset medium,
    CRF 23), not a quality-first setting.
    """
    ffmpeg_path = get_ffmpeg_path()
    print(f"  (using ffmpeg: {ffmpeg_path})")

    frame_iter = iter(frames)
    first_frame = next(frame_iter)
    width, height = first_frame.size
    command = [
        ffmpeg_path, "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(config.FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", config.VIDEO_PRESET,
        "-crf", str(config.VIDEO_CRF),
        "-pix_fmt", config.VIDEO_PIXEL_FORMAT,
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run_ffmpeg_writing_frames(command, first_frame, frame_iter)


def save_mp4_two_pass(frames, out_path, crf=14):
    """
    Saves frames as an .mp4 video file (H.264) via a lossless intermediate,
    for live wallpapers (see live_wallpaper.py) - re-rendered less often
    than the (already lengthy) ambient videos main.py produces, so a more
    careful export is worth it there.

    The (expensive, GPU-rendered) frame generator is only consumed once:
    frames are first streamed into a lossless intermediate file (FFV1 codec
    in a Matroska container) next to `out_path`, and the final x264 encode
    reads from that already-decoded intermediate - re-rendering the scene
    is far more expensive than a single extra encode pass. The intermediate
    is deleted afterwards.

    Despite the name (kept for call-site compatibility), the final encode
    is now a single CRF pass, not x264's classic two-pass ABR mode: the old version target-bitrate-capped this at a fairly
    modest LIVE_WALLPAPER_BITRATE_MBPS=8 for what can be a 4K 30-second
    loop, and x264's two-pass ABR mode exists specifically to hit a
    *bitrate target* as precisely as possible - it has nothing to offer
    when the actual goal is "as much quality as CRF wants to spend",
    which is exactly what this project's "quality over generation time"
    priority calls for. A single CRF pass at a low CRF value is the
    strictly-correct choice for that goal, and simpler besides.
    """
    ffmpeg_path = get_ffmpeg_path()
    print(f"  (using ffmpeg: {ffmpeg_path})")

    tmp_dir = out_path.parent / "_tmp_render"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # A forcibly-killed process (e.g. a crashed or manually terminated
    # render) skips the `finally` block below entirely, leaving a stale
    # leftover .mkv (or an empty directory) behind indefinitely - nothing
    # under this tmp_dir is ever read back across runs (see docstring:
    # "intermediate is deleted afterwards"), so it's always safe to purge
    # anything left over from a previous, non-clean exit before starting.
    for leftover in tmp_dir.iterdir():
        if leftover.is_file():
            leftover.unlink()
    intermediate_path = tmp_dir / f"{out_path.stem}_lossless.mkv"

    try:
        frame_iter = iter(frames)
        first_frame = next(frame_iter)
        width, height = first_frame.size

        print("  [encode] Rendering frames into a lossless intermediate ...")
        _run_ffmpeg_writing_frames(
            [
                ffmpeg_path, "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(config.FPS),
                "-i", "-",
                "-an",
                "-c:v", "ffv1",
                str(intermediate_path),
            ],
            first_frame, frame_iter,
        )

        print(f"  [encode] Final CRF {crf}/{config.VIDEO_PRESET} pass from the lossless intermediate ...")
        _run_ffmpeg([
            ffmpeg_path, "-y", "-loglevel", "error",
            "-i", str(intermediate_path),
            "-an", "-c:v", "libx264",
            "-preset", config.VIDEO_PRESET, "-crf", str(crf),
            "-pix_fmt", config.VIDEO_PIXEL_FORMAT, "-movflags", "+faststart",
            str(out_path),
        ])
    finally:
        intermediate_path.unlink(missing_ok=True)
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()


def _run_ffmpeg_writing_frames(command, first_frame, frame_iter):
    """Runs `command` (an ffmpeg invocation reading rawvideo from stdin) and streams `first_frame` + `frame_iter` into it."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    # ffmpeg writes to stderr (e.g. error messages) regardless of whether
    # we're still sending it data on stdin. Without reading stderr as it
    # comes in, on a separate thread, the pipe buffer could fill up and
    # block ffmpeg while it's trying to write to it - and then our process
    # would hang on stdin.write() below, because ffmpeg would stop reading
    # its input (a classic pipe deadlock, easier to hit the bigger the
    # frames are - and with large photos a single frame can be tens of MB).
    stderr_chunks = []
    stderr_thread = threading.Thread(target=lambda: stderr_chunks.append(process.stderr.read()))
    stderr_thread.start()

    # If ffmpeg dies mid-stream (bad codec args, disk full, etc.), writing
    # to its stdin raises BrokenPipeError/OSError - previously uncaught,
    # which crashed this function with that low-level pipe error instead of
    # ffmpeg's own, actually-useful stderr message (still sitting unread in
    # stderr_chunks). Catch it, then fall through to the same
    # close/wait/join/raise sequence as the clean-exit path below so the
    # real error surfaces either way.
    try:
        process.stdin.write(np.asarray(first_frame).tobytes())
        for frame in frame_iter:
            process.stdin.write(np.asarray(frame).tobytes())
    except (BrokenPipeError, OSError):
        pass

    try:
        process.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    process.wait()
    stderr_thread.join()

    if process.returncode != 0:
        error_output = b"".join(stderr_chunks).decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed:\n{error_output}")


def _run_ffmpeg(command):
    """Runs a plain ffmpeg command (no frame data piped through stdin) and raises with its stderr output on failure."""
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode(errors='ignore')}")
