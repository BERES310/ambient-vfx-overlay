"""
live_wallpaper.py

Exports a short, seamlessly looping .mp4 to output/live_wallpapers/, meant
to be used as an animated desktop background with third-party tools (e.g.
Wallpaper Engine) that accept a video file. This script does NOT touch your
actual Windows desktop background - it only produces the .mp4 file; setting
it as your wallpaper is a separate step you do yourself in whatever tool
you use for that.

Same effects/config as main.py (config.py - snow, rain, fog, flicker,
lightning), but tuned for this use case:
    - shorter, on a length/crossfade you control (LIVE_WALLPAPER_*)
    - the tail is crossfaded into the head so the loop has no visible seam
      (see vfx_engine/loop.py)
    - encoded via a lossless intermediate + a final CRF pass (see
      image_io.save_mp4_two_pass) so the expensive GPU render is never
      redone just to re-encode - worth the extra encoding time since these
      are re-rendered far less often than a one-off ambient video
    - MASK_MODE = "depth" (see vfx_engine/depth.py) is config.py's default
      for every output mode, not just this one, but it matters especially
      here: it makes falling snow/rain disappear behind foreground objects
      in the photo instead of floating flatly over everything, which is
      far more noticeable when you're going to stare at the loop all day
      as a wallpaper than for a one-off video

Set config.LIVE_PREVIEW_WINDOW = True to also open a window that plays the
exported loop right after saving it (see vfx_engine/preview_window.py).

Running (from inside the vfx_overlay/ folder):

    python live_wallpaper.py
"""

import csv
import time
from datetime import datetime

from PIL import Image

import config
from main import (
    any_effect_needs_mask, render_enabled_effects, resolve_base_frames,
    resolve_effect_suffix, resolve_flicker_mask, resolve_mask_frames,
    resolve_postprocess_params, resolve_segmentation,
)
from vfx_engine import (
    color, color_grade, compositor, frame_cache, image_io, lens_dirt, lighting, mask, presets, provenance,
    render_checkpoint, weather_grade,
)
from vfx_engine.gui_moods import fit_dimension
from vfx_engine.lighting import apply_ambient_drift, apply_flicker
from vfx_engine.lightning import apply_flash, build_strike_schedule, resolve_strike_geometry
from vfx_engine.loop import make_seamless_loop_resumable
from vfx_engine.postprocess import apply_postprocess
from vfx_engine.progress import report_frames, report_resume, report_stage, with_progress

# Per-run timing log ("add time measuring for how long it
# takes to run a whole run of live_wallpaper.py and append results to csv
# file") - one row per process_image() call, so both a plain `python
# live_wallpaper.py` run and any script that calls process_image() directly
# (e.g. run_weather_benchmark.py) get timed the same way, for free.
BENCHMARK_CSV_PATH = config.OUTPUT_DIR / "live_wallpaper_benchmark.csv"
_BENCHMARK_CSV_HEADER = ["timestamp", "photo", "preset_label", "effect_suffix", "duration_seconds", "output_path"]


def _log_benchmark_row(photo_name, preset_label, effect_suffix, duration_seconds, output_path):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not BENCHMARK_CSV_PATH.exists()
    with open(BENCHMARK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_BENCHMARK_CSV_HEADER)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            photo_name, preset_label, effect_suffix,
            f"{duration_seconds:.3f}", str(output_path),
        ])


def process_image(image_path, preset_label=None, out_path=None):
    """
    Processes one photo into a seamlessly looping live-wallpaper .mp4 in
    output/live_wallpapers/. Times the whole call and appends a row to
    BENCHMARK_CSV_PATH; preset_label is an optional human-readable tag
    (e.g. "rain_fog") recorded alongside the auto-derived effect suffix -
    useful when a caller (e.g. run_weather_benchmark.py) applies a config
    preset that resolve_effect_suffix() can't fully distinguish on its own
    (e.g. a "clear/sunny" preset with every suffix-contributing flag off).

    out_path: optional explicit destination (the GUI passes an auto-versioned
    name so a new render never overwrites an older video); default is
    <photo>_<effects>.mp4 in output/live_wallpapers/.

    A render interrupted part-way (Cancel, a crash, a closed window) leaves a
    frame-level checkpoint behind (vfx_engine/render_checkpoint.py); running
    the same photo with the same settings again continues from that frame.
    """
    print(f"Processing: {image_path.name}")
    _start_time = time.perf_counter()

    report_stage("loading", "Loading the photo")
    image = image_io.load_base_image(image_path)
    # Output size cap (config.LIVE_WALLPAPER_MAX_DIMENSION): resize the
    # photo itself so every later stage - depth, particles, encoding - works
    # at the smaller size and the render is proportionally faster.
    max_dimension = config.LIVE_WALLPAPER_MAX_DIMENSION
    if max_dimension is not None:
        capped_size = fit_dimension(image.size, max_dimension)
        if capped_size != image.size:
            image = image.resize(capped_size, Image.LANCZOS)
    output_size = image.size

    # Spatial supersampling - see main.py's process_image
    # for the full explanation; identical mechanism here.
    ssaa_factor = max(1, int(config.SUPERSAMPLE_FACTOR))
    if ssaa_factor > 1:
        render_size = (output_size[0] * ssaa_factor, output_size[1] * ssaa_factor)
        image = image.resize(render_size, Image.LANCZOS)
    size = image.size

    num_frames = round(config.LIVE_WALLPAPER_DURATION_SECONDS * config.FPS)
    crossfade_frames = round(config.LIVE_WALLPAPER_CROSSFADE_SECONDS * config.FPS)

    # Computed once and reused for both masking (MASK_MODE = "depth") and
    # particle depth-realism (config.PARTICLE_DEPTH_REALISM) - see main.py's
    # process_image, which does the same thing.
    depth_array = None
    depth_confidence = None
    if config.MASK_MODE == "depth" or config.PARTICLE_DEPTH_REALISM:
        report_stage("depth", "Understanding the photo - depth (the first run also loads the AI models)")
        from vfx_engine import depth as depth_module
        depth_array, depth_confidence = depth_module.estimate_depth(image, return_confidence=True)

    # AI scene segmentation - see main.py's resolve_segmentation/process_image,
    # which do the same thing.
    report_stage("segmentation", "Understanding the photo - sky, ground and water")
    segmentation = resolve_segmentation(image, depth_array, num_frames)

    # Learned illumination (the design notes(a)) - see main.py's
    # process_image for the full explanation; identical mechanism here.
    illumination_result = None
    if config.ENABLE_LEARNED_ILLUMINATION:
        try:
            from vfx_engine import illumination as illumination_module
            illumination_result = illumination_module.estimate_illumination(image)
        except RuntimeError as exc:
            print(f"  (learned illumination unavailable, falling back to heuristics - {exc})")

    # Base-photo color-grade unification pass - see main.py's process_image
    # for the full explanation; identical mechanism here.
    if config.COLOR_GRADE_UNIFY_ENABLED:
        image = color_grade.unify_color_grade(
            image, segmentation=segmentation, strength=config.COLOR_GRADE_UNIFY_STRENGTH,
        )
    if config.WEATHER_GRADE_STYLE:
        image = weather_grade.apply_weather_grade(image, config.WEATHER_GRADE_STYLE, config.WEATHER_GRADE_STRENGTH)

    # Frame-level checkpoint/resume - same mechanism and fingerprint fields as
    # main.py's process_image (source photo hash, output mode, the whole
    # config snapshot incl. size/quality/length, manual mask hashes, frame
    # count); see render_checkpoint.py and frame_cache.py.
    fingerprint = None
    start_frame = 0
    if num_frames > 1:
        fingerprint = render_checkpoint.build_fingerprint(
            image_path, provenance._file_content_hash(image_path), "live_wallpaper",
            provenance._snapshot_config(),
            {mask_path.name: provenance._file_content_hash(mask_path) for mask_path in provenance.resolve_mask_paths(image_path)},
            num_frames,
        )
        start_frame = render_checkpoint.load_checkpoint(fingerprint)
        if start_frame > 0:
            print(f"  Resuming render from frame {start_frame}/{num_frames} (checkpoint found)")
            report_resume(start_frame, num_frames)

    report_stage("weather", "Building the weather")
    shared_mask = mask.build_mask(image, image_path, depth_array=depth_array, segmentation=segmentation) if any_effect_needs_mask() else None
    mask_frames_factory = resolve_mask_frames(image, image_path, depth_array, num_frames, shared_mask, segmentation=segmentation, depth_confidence=depth_confidence, start_frame=start_frame)

    lightning_schedule = None
    lightning_geometry = None
    if config.LIGHTNING_ENABLED:
        lightning_params = config.LIGHTNING_PARAMS
        lightning_schedule = build_strike_schedule(
            num_frames, fps=config.FPS,
            frequency=lightning_params["frequency"], strike_frames=lightning_params["strike_frames"],
        )
        lightning_geometry = resolve_strike_geometry(
            lightning_schedule, size, depth_array=depth_array, strike_frames=lightning_params["strike_frames"],
        )

    effect_frame_iterables = render_enabled_effects(
        size, num_frames, mask_frames_factory, lightning_schedule, lightning_geometry,
        image=image, depth_array=depth_array, segmentation=segmentation, illumination_result=illumination_result,
        start_frame=start_frame,
    )
    overlay_frames = compositor.layer_effects(size, num_frames, effect_frame_iterables, start_frame=start_frame)
    # The camera drift path (see vfx_engine/camera.py) is itself an exact
    # closed loop over num_frames, so it composes cleanly with
    # make_seamless_loop's crossfade further below - the crossfade has
    # nothing to hide here, it's just acting as the safety net it's
    # meant to be for the other, non-periodic effects.
    base_frames = resolve_base_frames(image, num_frames, depth_array=depth_array, depth_confidence=depth_confidence, start_frame=start_frame)
    final_frames = compositor.composite_onto_base(base_frames, overlay_frames)

    if config.FLICKER_ENABLED:
        flicker_mask = resolve_flicker_mask(image, shared_mask)
        # Seeded from the photo so a resumed render flickers exactly like the
        # frames it already finished (see main.py's process_image).
        flicker_seed = lens_dirt.content_seed(image, salt="flicker")
        final_frames = apply_flicker(final_frames, num_frames, mask=flicker_mask, seed=flicker_seed, start_frame=start_frame, **config.FLICKER_PARAMS)

    if config.LIGHTNING_ENABLED:
        lightning_params = config.LIGHTNING_PARAMS
        final_frames = apply_flash(
            final_frames, lightning_schedule, lightning_geometry, intensity=lightning_params["intensity"],
            color=lightning_params["color"], strike_frames=lightning_params["strike_frames"],
            falloff_radius_frac=lightning_params["flash_falloff_radius_frac"],
            min_intensity_frac=lightning_params["flash_min_intensity_frac"],
            start_frame=start_frame,
        )

    if config.AMBIENT_DRIFT_ENABLED and num_frames > 1:
        drift_strength = config.AMBIENT_DRIFT_STRENGTH * lighting.resolve_ambient_drift_scale(image)
        final_frames = apply_ambient_drift(final_frames, num_frames, strength=drift_strength, start_frame=start_frame)

    final_frames = apply_postprocess(final_frames, **resolve_postprocess_params(image, num_frames, size), start_frame=start_frame)
    # Report/cancel on the raw stream: make_seamless_loop buffers its whole
    # crossfade head before yielding, so counting downstream would leave the
    # GUI's bar at 0 % (and Cancel unheard) for the first crossfade_frames.
    report_stage("frames", "Rendering frames")
    final_frames = report_frames(final_frames, num_frames, start_frame=start_frame)

    # Supersampling downsample - see main.py's process_image
    # for the full explanation. Done BEFORE the loop crossfade (both are
    # linear-light operations, so the order does not change the result): a
    # resumed render then reloads its head frames from the frame cache at
    # output resolution, which is what the blend needs.
    if ssaa_factor > 1:
        final_frames = (color.downsample_linear(frame, ssaa_factor) for frame in final_frames)

    def load_cached_head_frame(index):
        cached = frame_cache.load_frame(fingerprint, index)
        if cached is None:
            raise RuntimeError(
                f"the checkpoint says frame {index} was finished, but its cached copy is missing - "
                f"delete this render's checkpoint and start over"
            )
        return color.decode_image(cached)[:, :, :3]

    final_frames = make_seamless_loop_resumable(
        final_frames, num_frames, crossfade_frames, start_frame=start_frame, head_loader=load_cached_head_frame,
    )

    # End of the linear-light compositing/post-processing chain - encode back to gamma-encoded 8-bit PIL Images exactly ONCE,
    # here, right before saving.
    final_frames = image_io.encode_frames_to_images(final_frames)
    if fingerprint is not None:
        final_frames = frame_cache.resume_or_checkpoint_frames(final_frames, fingerprint, start_frame, num_frames)
    final_frames = with_progress(final_frames, num_frames, label="Rendering loop", emit_events=False)

    effect_suffix = resolve_effect_suffix()
    if out_path is None:
        stem = image_path.stem + (f"_{preset_label}" if preset_label else effect_suffix)
        out_path = config.LIVE_WALLPAPERS_DIR / f"{stem}.mp4"
    image_io.save_mp4_two_pass(final_frames, out_path, crf=config.LIVE_WALLPAPER_CRF)
    print(f"  Saved: {out_path}")

    # The mp4 is fully written, so the checkpoint and its frame cache are no
    # longer needed. Left out of a try/finally on purpose: an interrupted
    # render must keep them so the next run can resume.
    if fingerprint is not None:
        render_checkpoint.delete_checkpoint(fingerprint)
        frame_cache.clear_frame_cache(fingerprint)

    # Render provenance - see main.py's process_image for
    # the full explanation; identical mechanism here.
    provenance_path = provenance.save_provenance(out_path, image_path, num_frames, "mp4")
    print(f"  Saved: {provenance_path.name}")

    duration_seconds = time.perf_counter() - _start_time
    _log_benchmark_row(image_path.name, preset_label or "", effect_suffix, duration_seconds, out_path)
    print(f"  Run time: {duration_seconds:.1f}s (logged to {BENCHMARK_CSV_PATH.name})")

    if config.LIVE_PREVIEW_WINDOW:
        from vfx_engine.preview_window import play_loop

        def on_rerender(overrides):
            """Applies the preview window's control-panel choices and
            re-renders this same photo, without popping a second preview
            window (this callback's own play_loop call IS the preview)."""
            originals = {attr: getattr(config, attr) for attr in overrides}
            original_preview_window = config.LIVE_PREVIEW_WINDOW
            config.LIVE_PREVIEW_WINDOW = False
            try:
                for attr, value in overrides.items():
                    setattr(config, attr, value)
                new_out_path = process_image(image_path, preset_label=preset_label)
            finally:
                for attr, value in originals.items():
                    setattr(config, attr, value)
                config.LIVE_PREVIEW_WINDOW = original_preview_window
            return new_out_path, output_size[0], output_size[1], config.FPS

        def on_fast_preview(overrides):
            """Same override apply/restore pattern as on_rerender, but calls
            fast_preview.render_fast_preview() instead - a small in-memory
            frame list instead of a full mp4 render."""
            from fast_preview import render_fast_preview
            originals = {attr: getattr(config, attr) for attr in overrides}
            try:
                for attr, value in overrides.items():
                    setattr(config, attr, value)
                return render_fast_preview(image_path)
            finally:
                for attr, value in originals.items():
                    setattr(config, attr, value)

        print("  Opening live preview window (close it, or press Escape, to continue) ...")
        play_loop(
            out_path, output_size[0], output_size[1], config.FPS,
            window_title=f"Live Wallpaper Preview - {image_path.stem}",
            on_rerender=on_rerender, on_fast_preview=on_fast_preview,
        )

    return out_path


def main():
    presets.apply_preset(config.ACTIVE_PRESET)
    if config.ACTIVE_PRESET is not None:
        print(f"Using preset: {config.ACTIVE_PRESET}")

    config.make_sure_folders_exist()

    images = image_io.find_input_images(config.INPUT_DIR)
    if not images:
        print(f"No photos found in: {config.INPUT_DIR}")
        print("Drop .png/.jpg/.jpeg/.bmp/.webp files there and run the program again.")
        return

    for image_path in images:
        process_image(image_path)

    print(f"\nDone! Processed {len(images)} photo(s) -> {config.LIVE_WALLPAPERS_DIR}")


if __name__ == "__main__":
    main()
