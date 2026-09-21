"""
compositor.py

Two functions that combine effects (snow, fog, rain...) into one finished
image:

    overlay = layer_effects(size, num_frames, [snow_frames, fog_frames])
    final_frames = composite_onto_base(photo, overlay)

`layer_effects` stacks several transparent effects on top of each other
into a single layer, and `composite_onto_base` applies that layer onto the
real input photo.

Both now operate in LINEAR light, not gamma-encoded 8-bit -
alpha-compositing several bright, overlapping layers (rain + fog + god
rays + a lightning flash, say) is fundamentally an additive operation, and
doing it directly on sRGB-encoded values undershoots how bright the result
should actually look. `layer_effects` now yields linear float32 (height,
width, 4) numpy arrays instead of PIL Images - decoding each GPU/PIL-drawn
effect layer from its gamma-encoded 8-bit output exactly ONCE, right here,
is the entry point into this pipeline's linear-light compositing chain
(see color.py's module docstring for the full explanation and where that
chain ends - image_io.encode_linear_frames, right before saving).
"""

from PIL import Image

from vfx_engine import color


def layer_effects(size, num_frames, effect_frame_iterables, start_frame=0):
    """
    Generator - combines frame streams from several effects into a single
    sequence of linear-light RGBA numpy arrays (effects are layered on top
    of each other in the order given, via color.over - see that module),
    yielding one combined frame at a time. If no effects were given,
    yields a fully-transparent linear array per frame - so downstream code
    can always count on num_frames frames of a consistent type.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py).
        Only relevant to the empty-effects fallback branch below - when at
        least one real effect iterable is given, each of those iterables is
        assumed to already be start_frame-truncated by its own caller (the
        usual contract elsewhere in this codebase), so this function itself
        just zips them together frame-for-frame either way. The fallback
        branch has no such upstream iterable to inherit truncation from, so
        it skips frames 0..start_frame-1 itself, keeping its own output
        count (num_frames - start_frame) consistent with what a real,
        start_frame-aware effect stream would have yielded.
    """
    width, height = size
    if not effect_frame_iterables:
        empty = color.decode_image(Image.new("RGBA", size, (0, 0, 0, 0)))
        for _ in range(start_frame, num_frames):
            yield empty
        return

    for frames_at_step in zip(*effect_frame_iterables):
        combined = color.decode_image(frames_at_step[0])
        for frame in frames_at_step[1:]:
            combined = color.over(color.decode_image(frame), combined)
        yield combined


def composite_onto_base(base_image_or_frames, overlay_frames):
    """
    Generator - applies each frame of the combined effects onto the real
    input photo and yields the finished frames as linear float32 (height,
    width, 3) numpy arrays (RGB only - the base photo is opaque, so the
    composited result's alpha is always 1 and isn't carried further).

    base_image_or_frames: either a single static PIL Image (composited
        onto every overlay frame unchanged - the original behavior), or
        an iterable yielding one base frame per overlay frame (e.g.
        vfx_engine.camera.apply_camera_drift) - letting the photo itself
        slowly drift/zoom under the effects instead of staying perfectly
        pixel-locked. Either way, still a gamma-encoded 8-bit PIL Image -
        decoded to linear here, the same as overlay_frames already are
        (see layer_effects).
    """
    if isinstance(base_image_or_frames, Image.Image):
        base_linear = color.decode_image(base_image_or_frames)
        for overlay in overlay_frames:
            yield color.over(overlay, base_linear)[:, :, 0:3]
    else:
        for base_frame, overlay in zip(base_image_or_frames, overlay_frames):
            yield color.over(overlay, color.decode_image(base_frame))[:, :, 0:3]
