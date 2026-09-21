"""
segmentation.py

AI semantic scene segmentation - classifies WHAT every pixel of a photo
actually is (sky, water, vegetation, ground), as opposed to depth.py's
estimate_depth(), which only tells you HOW FAR it is. Several heuristics
elsewhere in this codebase currently guess "is this pixel ground/sky/
water" from depth + vertical position + brightness alone (see
ground_fx.build_ground_mask's docstring for a concrete example of where
that guess can go wrong: a lake sitting at "ground-like" depth/position in
the lower frame, or a splash landing on a raised bench instead of true
flat ground) - this module replaces that guess with a real per-pixel
classification from a pretrained segmentation model.

Model: SegFormer (nvidia/segformer-b5-finetuned-ade-640-640), the largest
published SegFormer checkpoint, trained on ADE20K's 150-class "scene
parsing" label set - see https://github.com/CSAILVision/sceneparsing.
Run once per photo (the result is cached and reused for every frame, the
same as depth.py's estimate_depth), so, exactly like the Large depth
checkpoint in depth.py, there's no real reason to trade quality for speed
here - generation time is not a constraint for this project.

this calls the model's raw SegformerForSemanticSegmentation
+ SegformerImageProcessor directly, NOT the high-level "image-segmentation"
pipeline this module originally used - two structural upgrades that both
needed that lower-level access:
  - 9.1 (soft/confidence-aware masks): the high-level pipeline collapses
    the model's real per-pixel class probabilities down to a hard binary
    mask (argmax'd, then thresholded) before this code ever saw them -
    genuine classifier uncertainty at an ambiguous boundary (a tree
    branch thinning against open sky, mist blurring a shoreline) was
    thrown away, not preserved. Calling the model directly exposes the
    raw per-class LOGITS, upsampled (bilinear, in one pass - see
    _infer_tile_probabilities) and softmaxed into genuine continuous
    per-pixel probabilities - a real soft, model-informed edge instead of
    a hard decision that gets blurred after the fact. This also
    supersedes.2's earlier NEAREST->BILINEAR fix (see
    _infer_tile_probabilities/estimate_segmentation below) - there is no
    longer a hard per-class mask to upsample at all; the softness is
    intrinsic to the probability field itself now, not a resize-filter
    choice applied on top of one.
  - 9.3 (higher effective resolution): the model's own working resolution
    (640x640, fixed) is far coarser than a modern output photo - feeding
    a whole large photo through in one pass means fine structure (leaf
    gaps, distant treeline texture) is already lost in the model's own
    internal downsampling before any resize step downstream even
    happens. estimate_segmentation now tiles large photos into
    overlapping ~640px windows (see _tile_boxes), runs full model
    inference on EACH tile at close to its native resolution, and blends
    the overlapping per-tile probability fields back together with a
    smooth 2D window (_tile_weight) - so each region of a large photo
    gets inference at close to 1:1 effective density instead of the
    whole photo being squeezed through one 640x640 pass regardless of
    its real size. Small photos (see _TILE_MIN_DIMENSION) skip tiling
    entirely and use one direct pass, matching the old single-pass
    behavior when tiling wouldn't actually help.

Same lazy-load/local-cache pattern as depth.py: model weights and the
HuggingFace/PyTorch cache are kept inside vfx_overlay/models/ (never the
user's global AppData/home-folder caches), and the process runs fully
offline once the model has been downloaded once - see the environment
variables below, set before torch/transformers get imported.
"""

import os

import config

_MODELS_DIR = config.MODELS_DIR
os.environ.setdefault("HF_HOME", str(_MODELS_DIR / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_MODELS_DIR / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(_MODELS_DIR / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# The model is expected to already be cached locally (see MODELS_DIR
# above) - force fully offline mode so no network call is ever made (not
# even a metadata/rate-limit check), the same as depth.py.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from vfx_engine import cache_store

_MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
_model = None
_processor = None

# tiling parameters. _TILE_SIZE matches the model's own
# native working resolution (640x640 - see this file's docstring) - a
# tile this size gets fed to the model at close to 1:1, instead of being
# downscaled further the way a whole large photo would be. Overlap is a
# fraction of _TILE_SIZE, blended via a smooth raised-cosine window (see
# _tile_weight) so adjacent tiles don't show a hard seam where they meet.
_TILE_SIZE = 640
_TILE_OVERLAP_FRACTION = 0.25
# Below this, in EITHER dimension... actually see _needs_tiling below -
# a photo no larger than this already fits a single _TILE_SIZE pass at
# close to 1:1 density, so tiling would only add seam-blending overhead
# for no real resolution gain - skip it and run one direct pass, exactly
# matching this function's pre-9.3 single-pass behavior for such photos.
_TILE_MIN_DIMENSION = 800

# ADE20K's 150 "scene parsing" class labels, as this specific checkpoint's
# own id2label config actually spells them (verified against the model's
# config.json, not guessed) - grouped into the coarse categories the rest
# of this pipeline consumes. Deliberately exact-match, not substring
# match: a naive substring check on the raw label text would, for
# example, wrongly match "streetlight" against a "tree" keyword (it
# literally contains the substring "tree" at index 1) or "pool table"
# against a "pool" keyword - exact-matching each pixel's single winning
# label against a fixed set of known-good ADE20K label strings has none
# of those false-positive traps.
_SKY_LABELS = {"sky"}
_WATER_LABELS = {"water", "sea", "river", "lake", "waterfall", "swimming pool"}
# these four ADE20K classes used to be flattened into one
# undifferentiated "vegetation" mask the instant this function returned,
# throwing away a real distinction the model itself already makes - a palm
# frond moves very differently in wind than an oak canopy, and a flower
# head bobs on a flexible stem more like a blade of grass than like a
# rigid branch. Split into three sway-relevant groups instead of one:
#   "tree"/"plant" - both woody/shrub-like enough that the existing
#       trunk-to-canopy height-graded sway model (vegetation.render_vegetation_sway)
#       still genuinely applies to both.
#   "palm" - kept separate: fronds whip/flutter rapidly around a crown
#       rather than a broad canopy gusting as one mass (see
#       vegetation.render_palm_sway).
#   "flower" - kept separate: no rigid trunk section at all, closer to
#       grass's low-stiffness traveling-wave bobbing than to a tree's
#       canopy-height model (see vegetation.render_grass_sway, reused
#       directly for this category too - see main.py's
#       render_vegetation_effects).
_VEGETATION_TREE_LABELS = {"tree", "plant"}
_VEGETATION_PALM_LABELS = {"palm"}
_VEGETATION_FLOWER_LABELS = {"flower"}
# "Hard" ground: paved/bare surfaces that plausibly show a wet sheen or a
# splash ring (see ground_fx.py) - deliberately excludes grass/field (see
# _GROUND_SOFT_LABELS below): real grass doesn't read as "wet" the same
# way stone/asphalt does, which is exactly the "not grass ... mistaken for
# [ground]" gap this module exists to close (see this file's docstring).
_GROUND_HARD_LABELS = {"floor", "road", "sidewalk", "earth", "path", "dirt track", "land", "sand", "runway"}
# "Soft" ground: still walkable/settleable terrain (snow accumulates here
# too), just not a hard reflective surface - kept as its own category so
# callers can choose whether to include it (see ground_fx.build_ground_mask's
# include_soft_ground param).
_GROUND_SOFT_LABELS = {"grass", "field"}
# "the model is already being run, fully, on every photo -
# mountain, hill, rock, ..., building, and dozens of other classes it's
# already classifying internally are computed and then discarded because
# nothing in _CATEGORY_LABELS references them." Two of the write-up's
# named examples are consumed below (see estimate_segmentation's rock/
# building-adjacency gating of ground_hard) - a bare rock face or a
# building's foundation line isn't real "hard ground" in the sense
# ground_fx.py's wet-sheen/splash effects mean, the same "this doesn't
# actually belong in the mask I gated on" failure shape as 9.5a's water-
# reflection case. Verified against this checkpoint's actual id2label
# (models/huggingface/hub/models--nvidia--segformer-b5-finetuned-ade-640-640/
# .../config.json) - not guessed, same standard as every other label set
# in this file.
_ROCK_TERRAIN_LABELS = {"mountain", "hill", "rock"}
_BUILDING_LABELS = {"building", "house", "skyscraper"}

_CATEGORY_LABELS = {
    "sky": _SKY_LABELS,
    "water": _WATER_LABELS,
    "vegetation_tree": _VEGETATION_TREE_LABELS,
    "vegetation_palm": _VEGETATION_PALM_LABELS,
    "vegetation_flower": _VEGETATION_FLOWER_LABELS,
    "ground_hard": _GROUND_HARD_LABELS,
    "ground_soft": _GROUND_SOFT_LABELS,
    "rock_terrain": _ROCK_TERRAIN_LABELS,
    "building": _BUILDING_LABELS,
}

# The three per-species vegetation categories above that all need the same
# water-reflection cross-gating
# and are collectively what this module's docstring/history calls
# "vegetation" - kept as one tuple so that gating loop and any future code
# needing "every vegetation-ish category" has one place to look, instead of
# each needing its own hardcoded list of the same three names.
_VEGETATION_CATEGORY_NAMES = ("vegetation_tree", "vegetation_palm", "vegetation_flower")

# How far (in pixels, at the photo's own resolution) a "vegetation" pixel
# must be from any "water" pixel to still count as vegetation - see the
# water-reflection cross-gating fix in estimate_segmentation below
#. A calm lake's reflection of trees/shoreline genuinely
# looks like tree texture at the pixel level, so the model classifies
# those reflection pixels as "tree"; a pure per-pixel water/vegetation
# overlap check would only catch a reflection landing EXACTLY on a water
# pixel, but segmentation boundaries are never pixel-perfect, so the
# exclusion is dilated outward by this many pixels to also catch
# reflection pixels the model placed just past the water mask's own edge.
#
# this used to be a flat 20px regardless of the photo's
# actual resolution, unlike depth.py's _edge_refine_radius, which
# explicitly scales with min(image_size) "so its neighborhood stays
# meaningful whether the photo is 720p or 8K." At this pipeline's
# routinely-cited 7000px+ supersampled working resolutions, a flat 20px
# band covers a far smaller fraction of the frame than it did at whatever
# resolution the constant was originally tuned against, letting real
# tree-reflection/terrain-boundary bleed extend past the exclusion zone.
# _resolve_adjacency_px below mirrors _edge_refine_radius's exact
# min(image_size)-scaling pattern, with a divisor of 350 chosen so this
# module's original, already-validated 20px floor holds unchanged up
# through min(image_size) == 7000 (the pipeline's own routinely-cited
# supersampled working resolution) and only grows proportionally above
# that, rather than needlessly perturbing behavior at resolutions this
# constant was already tuned and confirmed correct against.
_WATER_ADJACENCY_PX = 20

# Same idea as _WATER_ADJACENCY_PX, for the rock/building-adjacency
# gating of ground_hard - kept as its own named constant
# (independently tunable) even though it currently shares the same
# default value, rather than reusing _WATER_ADJACENCY_PX for an
# unrelated feature just because the numbers happen to match today.
_TERRAIN_ADJACENCY_PX = 20


def _resolve_adjacency_px(image_size, base_px):
    """
    scales an adjacency-exclusion radius (in pixels)
    by the photo's own resolution, the same min(image_size)-based approach
    depth.py's _edge_refine_radius already uses and has already proven
    correct for an analogous "keep a fixed pixel neighborhood meaningful
    across wildly different photo resolutions" need. Never returns less
    than `base_px` - a photo at or below the resolution the constant was
    originally tuned against keeps that exact original behavior; only
    higher resolutions get a proportionally larger band.
    """
    return max(base_px, round(min(image_size) / 350.0))



def _exclude_pixels_near(mask, exclusion_source_mask, adjacency_px):
    """
    Returns `mask` with any pixel within `adjacency_px` of a nonzero
    `exclusion_source_mask` pixel zeroed out - the shared "mask gates
    mask" pattern this module uses whenever one category's classification
    is known to be an unreliable proxy for another right at their
    boundary (originally.5a's water-reflection gating of
    vegetation; reused as-is for.5's rock/building-adjacency
    gating of ground_hard - same failure shape, same fix). Dilates
    exclusion_source_mask outward by adjacency_px so segmentation
    boundaries slightly past the raw exclusion mask's own edge are still
    caught, since a real model's classification boundary is never pixel-
    perfect. A no-op when exclusion_source_mask is None or entirely zero
    (nothing detected to exclude against).

    Uses cv2.dilate with a full square structuring element (a true max
    filter - the same operation this used to run via PIL's
    ImageFilter.MaxFilter) instead of PIL. PIL's MaxFilter is an
    unoptimized rank filter, roughly O(width * height * window_size^2) -
    at this project's real working resolution (SUPERSAMPLE_FACTOR-
    upscaled 4K-class photos routinely exceed 7000px on the long edge -
    see main.py) and this function's 41px window, a SINGLE call measured
    at 100+ seconds, called up to 5 times per photo (once per vegetation
    category against water, twice more for ground_hard against
    rock_terrain/building). THIS - not segmentation's model inference,
    which was independently verified fast in isolation - turned out to be
    the actual cause of this project's reported 10-minute live_wallpaper.py
    stalls, and of the low ("<30%") CPU usage reported alongside them: one
    CPU core pegged inside PIL's C loop for minutes while the GPU and
    every other core sat idle. cv2.dilate computes the identical
    max-in-window result via a separable, vectorized algorithm instead -
    verified bit-for-bit identical output against PIL's MaxFilter on real
    data, ~450x faster in testing (see diag_cv2_vs_pil_dilate.py).
    """
    if exclusion_source_mask is None or exclusion_source_mask.max() <= 0.0:
        return mask
    dilation_size = adjacency_px * 2 + 1   # match PIL MaxFilter's old odd-window-size convention
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
    nearby = cv2.dilate((exclusion_source_mask * 255.0).astype("uint8"), kernel)
    nearby = nearby.astype("float32") / 255.0
    return mask * (1.0 - nearby)


def _get_model_and_processor():
    """
    Lazily loads (and caches, both on disk in models/ and in memory for
    the rest of the run) the segmentation model's raw components directly
    - SegformerForSemanticSegmentation (the network itself) and
    SegformerImageProcessor (input preprocessing: resize-to-640x640,
    normalize) - instead of the high-level "image-segmentation" pipeline
    this module originally used (see this file's docstring, design notes
    9.1/9.3, for why direct access is needed).
    """
    global _model, _processor
    if _model is None:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [segmentation] Loading scene-segmentation model on {device} (first run downloads it into {_MODELS_DIR}) ...")
        _processor = SegformerImageProcessor.from_pretrained(_MODEL_NAME)
        _model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_NAME)
        _model.to(device)
        _model.eval()
    return _model, _processor


def _release_model():
    """
    Frees the segmentation model's GPU memory (moves it back to CPU and
    drops the cached model object) immediately after inference, the same
    way depth.py's _release_pipeline() already does for the depth model -
    see that function's docstring for the measured root cause (two large
    models sharing one small GPU's memory pool fragments PyTorch's CUDA
    caching allocator badly enough to force slow, effectively-synchronous
    cudaMalloc calls). depth.py releases its own model right after use,
    but this SegFormer-B5 model - loaded right after depth's, on the same
    GPU - had no equivalent release, so it stayed GPU-resident for the
    rest of the entire render (all 720+ per-frame postprocess passes),
    competing with postprocess.py's own per-frame allocations for the
    same small VRAM pool for the whole run instead of just its own,
    one-time-per-photo inference pass.

    Safe to call unconditionally: estimate_segmentation() is only called
    once per photo (its result is cached and reused for every frame - see
    main.py's process_image/live_wallpaper.py's process_image), so the
    model is never needed again for THIS photo; a later photo in the same
    run reloads it from the on-disk cache in _get_model_and_processor()
    (a couple seconds, not a re-download).
    """
    global _model, _processor
    if _model is not None:
        import gc

        import torch
        _model.to("cpu")
        _model = None
        _processor = None
        # gc.collect() right after dropping the reference (see depth.py's
        # _release_pipeline docstring for the measured reason) - without
        # it, this model can stay alive on the CPU heap past this function
        # returning, stacking on top of the NEXT model's load instead of
        # actually being freed first.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _tile_boxes(width, height, tile_size, stride):
    """
    Yields (left, top, right, bottom) crop boxes covering the full
    width x height image with overlap -. Each row/column of
    starts is spaced by `stride` (< tile_size, so adjacent tiles
    overlap), with one final start forced flush against the image's own
    right/bottom edge if the regular stride spacing would otherwise
    overshoot or leave a gap short of it - guarantees full coverage with
    no photo-edge sliver left untiled, without needing a tile smaller
    than tile_size just to make the arithmetic land exactly.
    """
    def _starts(total, size):
        if total <= size:
            return [0]
        starts = list(range(0, total - size + 1, stride))
        if starts[-1] != total - size:
            starts.append(total - size)
        return starts

    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    for y in _starts(height, tile_h):
        for x in _starts(width, tile_w):
            yield (x, y, x + tile_w, y + tile_h)


def _tile_weight(tile_w, tile_h, overlap, box, width, height):
    """
    Returns a (tile_h, tile_w) float32 blend weight for one tile - 1.0 in
    the tile's interior, tapering smoothly (raised-cosine, i.e. a Hann-
    style window) to 0 over `overlap` pixels at each INTERNAL edge (one
    actually shared with a neighboring tile - see _tile_boxes), so
    overlapping tiles blend into each other smoothly instead of showing a
    hard seam where their probability fields disagree. An edge that
    coincides with the photo's own true boundary is NOT tapered (full
    weight all the way to that edge) - there is no neighboring tile past
    the photo's edge to pick up the slack, so tapering there would just
    darken/zero-weight real edge content for no reason.
    """
    left, top, right, bottom = box
    x_weight = np.ones(tile_w, dtype="float32")
    y_weight = np.ones(tile_h, dtype="float32")

    if overlap > 0:
        ramp_up = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap, dtype="float32"))
        if left > 0:
            x_weight[:overlap] = np.minimum(x_weight[:overlap], ramp_up)
        if right < width:
            x_weight[-overlap:] = np.minimum(x_weight[-overlap:], ramp_up[::-1])
        if top > 0:
            y_weight[:overlap] = np.minimum(y_weight[:overlap], ramp_up)
        if bottom < height:
            y_weight[-overlap:] = np.minimum(y_weight[-overlap:], ramp_up[::-1])

    return y_weight[:, None] * x_weight[None, :]


def _infer_tile_probabilities(model, processor, tile_image):
    """
    Runs raw model inference on one image tile (any pixel size) and
    returns a (num_labels, tile_height, tile_width) float32 array of
    genuine per-pixel class PROBABILITIES at the tile's own true pixel
    resolution -.

    The processor resizes (stretches, not aspect-preserving) whatever
    tile it's given to the model's fixed 640x640 input size; the model's
    decode head then outputs logits at a QUARTER of that (160x160) - see
    this file's docstring. Those raw logits are bilinear-upsampled
    DIRECTLY to the tile's own true (tile_height, tile_width) in one
    pass (not first to 640x640 then again to the tile size - that would
    be exactly the kind of double-resample fixed elsewhere),
    which also automatically un-does the processor's own initial
    stretch-to-square (whatever aspect distortion the input resize
    introduced, resizing the output logits back to the tile's own
    original width/height exactly reverses it geometrically - a tile
    doesn't need to be square). Softmax is applied AFTER upsampling, not
    before, matching how transformers' own reference post-processing
    handles this (interpolating raw logits, then arg-deciding) - so the
    result is a real, per-output-pixel probability distribution (sums to
    1.0 across the class dimension at every pixel), not an upsampled
    blend of already-discrete decisions.

    (A batched variant of this function was tried as a performance fix
    and reverted - measured to make things WORSE, not better, on this
    project's target hardware: the design notes' Progress Log
    for the investigation. The real bottleneck turned out to be
    elsewhere - see this file's docstring's "GPU memory contention"
    section.)
    """
    import torch
    import torch.nn.functional as F

    tile_w, tile_h = tile_image.size
    device = next(model.parameters()).device
    inputs = processor(images=tile_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits   # (1, num_labels, h', w') - h'/w' = 640/4 = 160 for this checkpoint
        upsampled = F.interpolate(logits, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        probabilities = torch.softmax(upsampled, dim=1)

    return probabilities[0].to("cpu").numpy().astype("float32")


def estimate_segmentation(image):
    """
    Returns a dict of numpy float32 (height, width) arrays, each 0.0-1.0,
    same size/orientation as `image` (a PIL image) - one array per
    category in _CATEGORY_LABELS ("sky", "water", "vegetation_tree",
    "vegetation_palm", "vegetation_flower", "ground_hard", "ground_soft",
    "rock_terrain", "building"). Each value is a genuine per-pixel
    PROBABILITY - not a hard binary in/out decision. A
    confident interior pixel still reads close to 1.0 (softmax saturates
    hard when one class's logit clearly dominates), but a genuinely
    ambiguous boundary pixel (a thinning tree branch against open sky, a
    misty shoreline) now correctly reads as an intermediate value
    reflecting the model's own real uncertainty there, instead of being
    forced to a hard 0 or 1 and then papered over with a generic blur
    after the fact.

    for photos larger than _TILE_MIN_DIMENSION in either
    dimension, this tiles the photo into overlapping ~_TILE_SIZE windows
    (see _tile_boxes), runs full model inference on EACH tile near its
    native resolution, and blends the overlapping per-tile probability
    fields back together with a smooth window (_tile_weight) - so a large
    photo gets inference at close to 1:1 effective density everywhere,
    not the whole frame squeezed through one fixed 640x640 pass. Smaller
    photos use a single direct pass (tiling wouldn't add any resolution
    there, just seam-blending overhead for nothing).

    vegetation is split into THREE separate categories
    (vegetation_tree = "tree"/"plant", vegetation_palm = "palm",
    vegetation_flower = "flower") instead of one combined "vegetation"
    mask - each moves in wind differently enough (see
    vfx_engine/vegetation.py's render_vegetation_sway/render_palm_sway/
    render_grass_sway - the last one reused directly for flowers too) that
    flattening them into one mask, as this function used to, threw away a
    real distinction the model itself already makes per pixel.
    """
    # Resumable on-disk cache (see vfx_engine/cache_store.py): unlike
    # depth.py, this function's output depends only on the photo's pixel
    # content, not on any config.py setting (_CATEGORY_LABELS and the
    # tiling parameters above are fixed constants, not tunables) - so the
    # cache key is the image content hash alone. A hit skips loading the
    # SegFormer-B5 model entirely, not just the tiled inference loop -
    # exactly the multi-minute cost the resumable-cache feature exists to
    # let an interrupted/re-run batch skip.
    cache_key = cache_store.build_cache_key(image, "segmentation", version=1)
    cached = cache_store.cache_get("segmentation", cache_key)
    if cached is not None:
        return {category: cached[category] for category in _CATEGORY_LABELS}

    model, processor = _get_model_and_processor()
    width, height = image.size

    stride = max(1, round(_TILE_SIZE * (1.0 - _TILE_OVERLAP_FRACTION)))
    overlap = _TILE_SIZE - stride
    needs_tiling = max(width, height) > _TILE_MIN_DIMENSION
    boxes = list(_tile_boxes(width, height, _TILE_SIZE, stride)) if needs_tiling else [(0, 0, width, height)]

    # Reduce each tile's full (150-class) probability tensor down to just
    # this pipeline's ~9 CATEGORIES immediately, before accumulating
    # across tiles - keeping a full (150, height, width) buffer for the
    # whole photo would be enormous at this project's real working sizes
    # (SUPERSAMPLE_FACTOR-upscaled 4K-class photos routinely exceed
    # 7000px on the long edge - see main.py's SSAA block - which would be
    # tens of gigabytes at 150 channels). Accumulating only the ~9
    # categories this module actually needs cuts that by roughly the same
    # ~150/9 factor, and is exactly the same MAX-over-constituent-class-
    # indices reduction estimate_segmentation always ended with anyway.
    label2id = model.config.label2id
    category_class_indices = {
        category: [label2id[label] for label in labels if label in label2id]
        for category, labels in _CATEGORY_LABELS.items()
    }

    probability_sum = {category: np.zeros((height, width), dtype="float32") for category in _CATEGORY_LABELS}
    weight_sum = np.zeros((height, width), dtype="float32")

    def _accumulate_tile(tile_probabilities, box):
        # The CPU-only half of one tile's work (reduce 150 classes down to
        # this pipeline's ~9 categories, then add into the shared
        # probability_sum/weight_sum arrays) - runs on a background thread
        # so it overlaps with the GPU inference call for the NEXT tile
        # (see the main loop below) instead of the two waiting on each
        # other, which is what left the CPU idle roughly half the time
        # this loop was running. accumulate_executor below has exactly ONE
        # worker, so these calls still execute strictly in the same order
        # the sequential loop always submitted them in - the float
        # addition order into probability_sum/weight_sum is unchanged, so
        # the result is bit-exact identical to the old sequential version,
        # not just numerically close.
        left, top, right, bottom = box
        tile_w, tile_h = right - left, bottom - top
        tile_weight = _tile_weight(tile_w, tile_h, overlap, box, width, height) if needs_tiling else np.ones((tile_h, tile_w), dtype="float32")
        for category, class_indices in category_class_indices.items():
            if not class_indices:
                continue
            tile_category_values = tile_probabilities[class_indices].max(axis=0)
            probability_sum[category][top:bottom, left:right] += tile_category_values * tile_weight
        weight_sum[top:bottom, left:right] += tile_weight

    # Two small background thread pools overlap this loop's CPU-only work
    # with its GPU-only work instead of running everything on one thread
    # in strict lockstep (crop -> wait for GPU -> accumulate -> crop ->
    # wait for GPU -> ...), which is what kept both the CPU and GPU sitting
    # idle roughly half the time each:
    #   - crop_executor prefetches the NEXT tile's crop while the GPU is
    #     still busy on the CURRENT tile, so it's already in hand the
    #     instant it's needed (image.crop is a plain CPU/PIL op, entirely
    #     independent of the model and of every other tile).
    #   - accumulate_executor takes the CURRENT tile's finished
    #     probabilities and does the reduce+add step (see _accumulate_tile
    #     above) WHILE the main thread has already moved on to the GPU
    #     forward pass for the NEXT tile, instead of blocking that call.
    # Neither changes what gets computed or in what order - only when the
    # CPU happens to be doing it relative to the GPU - so output is
    # unchanged (see _accumulate_tile's docstring) while wall-clock time
    # and CPU utilization both improve.
    with ThreadPoolExecutor(max_workers=2) as crop_executor, ThreadPoolExecutor(max_workers=1) as accumulate_executor:
        crop_futures = {0: crop_executor.submit(image.crop, boxes[0])} if boxes else {}
        pending_accumulate = []
        for i, box in enumerate(boxes):
            tile_image = crop_futures.pop(i).result()
            if i + 1 < len(boxes):
                crop_futures[i + 1] = crop_executor.submit(image.crop, boxes[i + 1])

            tile_probabilities = _infer_tile_probabilities(model, processor, tile_image)   # (num_labels, tile_h, tile_w)
            pending_accumulate.append(accumulate_executor.submit(_accumulate_tile, tile_probabilities, box))

        for future in pending_accumulate:
            future.result()   # surface any exception, and make sure every tile is folded in before we read the arrays below

    safe_weight = np.clip(weight_sum, 1e-6, None)
    category_arrays = {category: values / safe_weight for category, values in probability_sum.items()}

    # Confirmed bug fix: water reflections of trees/shore
    # genuinely look like tree texture at the pixel level, so the model
    # classifies those reflection pixels as "tree" - vegetation.py then
    # correctly applies wind-sway to whatever mask it's handed, which is
    # why a coherent, correctly-swaying green shape can appear floating
    # over open water. Gate EACH of the three per-species vegetation masks
    #
    # against the water mask the same segmentation pass already produced -
    # exclude any pixel within _WATER_ADJACENCY_PX of a water pixel from
    # ever counting as vegetation, regardless of what the raw per-pixel
    # classification says.
    water_adjacency_px = _resolve_adjacency_px(image.size, _WATER_ADJACENCY_PX)
    water_mask = category_arrays.get("water")
    for vegetation_category in _VEGETATION_CATEGORY_NAMES:
        category_arrays[vegetation_category] = _exclude_pixels_near(
            category_arrays[vegetation_category], water_mask, water_adjacency_px,
        )

    # "a rock/mountain category could inform where NOT to
    # place ... splash-appropriate ground effects (a bare rock face isn't
    # 'hard ground' in the sense ground_fx.py means it)" - a rocky/
    # mountainous slope right at the boundary of a flatter bare area
    # sometimes gets classified as the generic "earth"/"land"/"path"
    # labels ground_hard already includes, rather than the more specific
    # "rock"/"mountain"/"hill" - exclude ground_hard pixels adjacent to a
    # detected rock/mountain region the same way water-adjacency excludes
    # reflections above: a jagged natural rock face doesn't show a
    # realistic wet-sheen highlight or splash ring the way real pavement
    # does. Same reasoning for "a building category could keep rain-
    # streak/wet-sheen placement off masonry differently than off natural
    # stone" - a building's foundation/base line is the same kind of
    # ambiguous boundary the model can misclassify as ground_hard's bare-
    # earth labels.
    terrain_adjacency_px = _resolve_adjacency_px(image.size, _TERRAIN_ADJACENCY_PX)
    category_arrays["ground_hard"] = _exclude_pixels_near(
        category_arrays["ground_hard"], category_arrays.get("rock_terrain"), terrain_adjacency_px,
    )
    category_arrays["ground_hard"] = _exclude_pixels_near(
        category_arrays["ground_hard"], category_arrays.get("building"), terrain_adjacency_px,
    )

    cache_store.cache_put("segmentation", cache_key, **category_arrays)

    # Release the model's GPU memory now (see _release_model's docstring) -
    # everything below this point (main.py/live_wallpaper.py's per-frame
    # effect rendering) is plain numpy/PIL work on these already-finished
    # arrays, the segmentation model is never touched again for this photo.
    _release_model()

    return category_arrays
