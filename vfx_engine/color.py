"""
color.py

Linear-light color space conversions - the boundary between
"gamma-encoded 8-bit PIL Images" (what every GPU shader/PIL-drawn effect
layer produces, and what the final saved file must be) and "linear
float32 numpy arrays" (what every COMPOSITING/BLENDING operation between
those layers should actually operate on).

Why this matters: light physically adds in LINEAR light, not in
gamma-encoded values. Alpha-compositing two layers, screen-blending a
flicker, or blurring/thresholding for bloom are all fundamentally additive
operations - doing them directly on sRGB-encoded 0-255 values (as this
pipeline used to, at every single stage) is mathematically wrong and
visibly undershoots how bright overlapping bright layers should look
(confirmed directly in the old code: compositor.py's Image.alpha_composite,
postprocess.py's bloom/tonemap math, lighting.py's ImageChops.screen
flicker blend, and every GPU framebuffer read (`fbo.read(dtype="f1")`)
were all sRGB-space, 8-bit, at every stage - a separate round-trip each
time, throwing away precision before the next stage even got to use it).

Scope: this fixes the CPU-side compositing/post-processing CHAIN - once an
effect layer leaves the GPU (or a PIL-drawn generator like lightning.py),
it's decoded to linear light exactly ONCE, right where compositor.py first
touches it, and stays in linear float32 through every subsequent stage
(layer stacking, base compositing, flicker, ambient drift, the lightning
flash, bloom, chromatic aberration, motion blur, tonemap) - only encoded
back to gamma-encoded 8-bit at the very end, right before the file is
saved. This does NOT make every individual GPU shader's own internal math
gamma-aware (that would mean decoding every texture sample and re-encoding
every frag_color write, in every .glsl file) - that's a much larger,
separate undertaking; this fixes the compositing/blending chain the
design notes's own examples are all drawn from.

Part 21 (GPU-resident pipeline, design notes "out of scope" section - explicitly
requested by the user anyway): everything downstream of the GPU particle/
fog/scene-warp shaders (compositor -> lighting/lightning -> postprocess)
used to round-trip through plain CPU numpy arrays, one conversion per
stage, discarding whatever device residency the upstream shader output
had. `decode_image` now returns a torch tensor resident on `resolve_device()`
(CUDA if available, else CPU) instead of a numpy array, and every function
below that previously branched on "numpy, big enough to be worth a torch
conversion, or numpy, small" gains a THIRD branch: "already a torch
tensor - compute directly on whatever device it's already on, no
conversion at all." This means a frame decoded once, on CUDA, stays a CUDA
tensor through the entire compositor/lighting/lightning/postprocess chain
without ever touching the CPU, until image_io.encode_frames_to_images's
single call to encode_image at the very end. The two original numpy
branches (large-array-via-CPU-torch, and small-array-plain-numpy) are left
completely unchanged, so every existing caller that still passes numpy
(tests, scalar colors, particles.py's internal sub-frame averaging) keeps
working exactly as before.
"""

import os

import numpy as np
import torch
from PIL import Image

_DEVICE = None


def resolve_device():
    """
    Returns the torch.device the GPU-resident compositing chain (Part 21)
    should run on - CUDA if available (mirrors depth.py/segmentation.py/
    surface_normals.py's own `"cuda" if torch.cuda.is_available() else
    "cpu"` device-detection pattern), else CPU. Resolved once and cached -
    torch.cuda.is_available() is cheap but not free, and this is called
    once per decoded frame.
    """
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

# The per-frame compositing chain below (srgb_to_linear/linear_to_srgb's
# fractional power, and over()'s several elementwise passes) is run ~13-17
# times per frame - once per effect layer - on multi-megapixel arrays. Plain
# numpy ufuncs are single-threaded, so this work only ever touched one of
# this machine's cores. torch's CPU tensor ops are vectorized AND
# multithreaded across all available cores by default, for bit-for-bit
# (float32-rounding-level) identical results - so routing the hot,
# large-array path through torch turns idle cores into real speedup instead
# of lowering output quality. Small/scalar inputs (e.g. lighting.py's single
# RGB color, or tests) fall back to the original plain-numpy formula, both
# to avoid tensor-conversion overhead not worth paying at that size and to
# keep this function's long-standing "works on scalars too" behavior intact.
torch.set_num_threads(os.cpu_count() or 4)
_TORCH_THRESHOLD = 50_000  # element count above which torch is worth the conversion overhead


def srgb_to_linear(channel):
    """Decodes an sRGB-encoded value (0.0-1.0 float) to linear light. Vectorized - works on scalars, numpy arrays, or (Part 21) torch tensors already resident on some device."""
    if isinstance(channel, torch.Tensor):
        return torch.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)
    if isinstance(channel, np.ndarray) and channel.size >= _TORCH_THRESHOLD:
        t = torch.from_numpy(np.ascontiguousarray(channel))
        out = torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)
        return out.numpy()
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(channel):
    """Encodes a linear-light value (0.0-1.0 float, CALLER must clip first - out-of-range input produces nonsense/NaN) back to sRGB. Also accepts (Part 21) torch tensors, computed directly on their own device."""
    if isinstance(channel, torch.Tensor):
        return torch.where(channel <= 0.0031308, channel * 12.92, 1.055 * (channel ** (1.0 / 2.4)) - 0.055)
    if isinstance(channel, np.ndarray) and channel.size >= _TORCH_THRESHOLD:
        t = torch.from_numpy(np.ascontiguousarray(channel))
        out = torch.where(t <= 0.0031308, t * 12.92, 1.055 * (t ** (1.0 / 2.4)) - 0.055)
        return out.numpy()
    return np.where(channel <= 0.0031308, channel * 12.92, 1.055 * (channel ** (1.0 / 2.4)) - 0.055)


def decode_image(image, device=None):
    """
    Converts a gamma-encoded 8-bit PIL Image (RGB or RGBA) to a linear
    float32 tensor, shape (height, width, 4) - RGB channels decoded to
    linear light, alpha (added as 1.0 if the source had none) left
    UNCHANGED (alpha is already a linear coverage/opacity value, not a
    gamma-encoded light intensity - only color channels need decoding).

    Part 21 (GPU-resident pipeline): returns a torch tensor resident on
    `device` (default: `resolve_device()`, i.e. CUDA if available) rather
    than a numpy array, so it can be handed straight to compositor.py/
    lighting.py/postprocess.py without ever touching the CPU. `device` is
    exposed as a parameter (rather than always resolving internally) so
    tests/scratch-verification scripts can force CPU-only decoding for
    numeric comparison against the pre-Part-21 numpy path.
    """
    if device is None:
        device = resolve_device()
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    pixels = torch.from_numpy(np.asarray(image, dtype="float32")).to(device) / 255.0
    rgb = srgb_to_linear(pixels[:, :, 0:3])
    alpha = pixels[:, :, 3:4]
    return torch.cat([rgb, alpha], dim=2)


def encode_image(linear_array, mode="RGB"):
    """
    Converts a linear float32 array (shape (height, width, 3) or
    (height, width, 4)) back to a gamma-encoded 8-bit PIL Image. Values are
    clipped to 0.0-1.0 first (this is the point in the pipeline where any
    additive headroom above 1.0 - from stacked bright layers, bloom, etc -
    must have already been resolved by a tonemap curve; clipping here is a
    safety net, not the intended way to handle over-range values).

    mode: "RGB" (default - the array's alpha channel, if any, is dropped)
        or "RGBA" (alpha is passed through as-is, unencoded, and included).

    Part 21: accepts either a torch tensor (any device - this is the final
    exit point from the GPU-resident chain, so the tensor is moved to CPU
    here) or a plain numpy array (unchanged path, used by callers - e.g.
    particles.py's internal sub-frame averaging, tests - that still pass
    numpy directly).
    """
    if isinstance(linear_array, torch.Tensor):
        rgb = linear_to_srgb(torch.clamp(linear_array[:, :, 0:3], 0.0, 1.0))
        if mode == "RGBA":
            alpha = torch.clamp(linear_array[:, :, 3:4], 0.0, 1.0)
            pixels = torch.cat([rgb, alpha], dim=2)
        else:
            pixels = rgb
        pixels_uint8 = (pixels * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
        return Image.fromarray(pixels_uint8, mode=mode)

    rgb = linear_to_srgb(np.clip(linear_array[:, :, 0:3], 0.0, 1.0))
    if mode == "RGBA":
        alpha = np.clip(linear_array[:, :, 3:4], 0.0, 1.0)
        pixels = np.concatenate([rgb, alpha], axis=2)
    else:
        pixels = rgb
    return Image.fromarray((pixels * 255.0 + 0.5).astype("uint8"), mode=mode)


def downsample_linear(array, factor):
    """
    Box-filter downsamples a linear float32 (height, width, channels)
    array by an exact integer `factor` in both dimensions -.4's
    spatial supersampling: averaging `factor` x `factor` blocks of pixels
    is the mathematically correct way to anti-alias a higher-resolution
    render down to its final size (equivalent to area-sampling each output
    pixel), and doing it here in LINEAR light (not on gamma-encoded
    values, and not via PIL - see this module's docstring for why PIL's
    filters can't be used on this pipeline's unclamped float32 data)
    avoids yet another needless precision round-trip.

    array's height/width MUST be exact multiples of `factor` (guaranteed
    by construction - see main.py, which builds the supersampled working
    resolution as output_size * factor, never an independently-rounded
    value that could leave a remainder).

    Part 21: `array` may be a torch tensor (main.py/live_wallpaper.py call
    this directly on the frame pulled out of the generator chain, which is
    now tensor-typed) or a plain numpy array - both are handled, computed
    on whatever device a tensor input is already on.
    """
    if factor <= 1:
        return array
    height, width = array.shape[0], array.shape[1]
    out_height, out_width = height // factor, width // factor
    if isinstance(array, torch.Tensor):
        reshaped = array.reshape(out_height, factor, out_width, factor, array.shape[2])
        return reshaped.mean(dim=(1, 3))
    reshaped = array.reshape(out_height, factor, out_width, factor, array.shape[2])
    return reshaped.mean(axis=(1, 3))


def over(foreground, background):
    """
    Standard alpha "over" compositing operator, in LINEAR light - both
    inputs are (height, width, 4) linear float32 arrays (RGB linear, alpha
    0.0-1.0). Returns a linear (height, width, 4) array: foreground drawn
    on top of background, with foreground's own alpha controlling the mix
    and the result's own alpha correctly accumulated (so stacking several
    partially-transparent layers via repeated `over()` calls composites
    correctly, not just the final onto-opaque-base case).

    Part 21: if `foreground`/`background` are already torch tensors (the
    normal case now that decode_image returns tensors), computes directly
    on whatever device they're already resident on - no CPU round-trip at
    all, unlike the pre-existing "large numpy array -> CPU torch -> numpy"
    branch below (which still exists, unchanged, for callers that pass
    plain numpy).
    """
    if isinstance(foreground, torch.Tensor):
        fg_rgb, fg_a = foreground[:, :, 0:3], foreground[:, :, 3:4]
        bg_rgb, bg_a = background[:, :, 0:3], background[:, :, 3:4]

        out_a = fg_a + bg_a * (1.0 - fg_a)
        safe_out_a = torch.where(out_a > 1e-6, out_a, torch.ones_like(out_a))
        out_rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / safe_out_a

        return torch.cat([out_rgb, out_a], dim=2)

    if foreground.size >= _TORCH_THRESHOLD:
        fg = torch.from_numpy(np.ascontiguousarray(foreground))
        bg = torch.from_numpy(np.ascontiguousarray(background))
        fg_rgb, fg_a = fg[:, :, 0:3], fg[:, :, 3:4]
        bg_rgb, bg_a = bg[:, :, 0:3], bg[:, :, 3:4]

        out_a = fg_a + bg_a * (1.0 - fg_a)
        safe_out_a = torch.where(out_a > 1e-6, out_a, torch.ones_like(out_a))
        out_rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / safe_out_a

        return torch.cat([out_rgb, out_a], dim=2).numpy()

    fg_rgb, fg_a = foreground[:, :, 0:3], foreground[:, :, 3:4]
    bg_rgb, bg_a = background[:, :, 0:3], background[:, :, 3:4]

    out_a = fg_a + bg_a * (1.0 - fg_a)
    safe_out_a = np.where(out_a > 1e-6, out_a, 1.0)   # avoid dividing by ~0 where both layers are fully transparent
    out_rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / safe_out_a

    return np.concatenate([out_rgb, out_a], axis=2)
