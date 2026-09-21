"""
loop.py

Makes a frame sequence loop seamlessly, regardless of how the frames were
generated - the standard technique for looping procedural/stochastic
animation (used for tileable game textures, ambient loops, etc.): render a
few extra frames past the target length, then crossfade that extra tail
back into the beginning, so frame N blends smoothly into frame 0 instead of
jump-cutting.

This works even though our particle motion (curl-noise wind, per-particle
random depth/speed) isn't naturally periodic - the crossfade hides the seam
in the pixels themselves, after the fact, so it doesn't matter how the
frames were produced.

Operates on linear-light float32 (height, width, 3) numpy arrays, the same representation compositor.py/postprocess.py/lighting.py use
- a crossfade is itself a blend, and this pipeline's linear-light chain
doesn't end until image_io.encode_frames_to_images runs, downstream of
this function (see main.py/live_wallpaper.py's call order).
"""

from collections import deque


def make_seamless_loop(frames, total_frames, crossfade_frames):
    """
    Generator - consumes exactly `total_frames` frames from `frames` and
    yields all of them back unchanged, except the last `crossfade_frames`,
    which get blended towards the first `crossfade_frames` frames (ramping
    from 0% to 100%) - so that looping the output straight back to its own
    start has no visible jump. The trade-off: the tail's last
    `crossfade_frames` become a blend rather than fully independent motion,
    which is unnoticeable as long as crossfade_frames stays small relative
    to total_frames (a second or so out of a 10-30s loop, say).

    crossfade_frames=0 disables this (frames pass through unchanged).
    """
    frame_iter = iter(frames)

    if crossfade_frames <= 0:
        for _ in range(total_frames):
            yield next(frame_iter)
        return

    max_crossfade = total_frames // 2
    if crossfade_frames > max_crossfade:
        print(
            f"  (loop.py: crossfade_frames={crossfade_frames} exceeds half of "
            f"total_frames={total_frames}; clamping to {max_crossfade} so the "
            f"head/middle/tail split stays valid)"
        )
        crossfade_frames = max_crossfade

    if crossfade_frames <= 0:
        for _ in range(total_frames):
            yield next(frame_iter)
        return

    head = [next(frame_iter) for _ in range(crossfade_frames)]
    yield from head

    middle_count = total_frames - 2 * crossfade_frames
    for _ in range(middle_count):
        yield next(frame_iter)

    tail = deque(maxlen=crossfade_frames)
    for _ in range(crossfade_frames):
        tail.append(next(frame_iter))

    for index, tail_frame in enumerate(tail):
        alpha = (index + 1) / (len(tail) + 1)   # ramps 0 -> 1 across the crossfade
        yield tail_frame * (1.0 - alpha) + head[index] * alpha
