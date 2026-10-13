"""
config.py

All program settings live in one place. Change values here - main.py never
needs to be touched to change what the program does.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders - everything stays inside vfx_overlay/
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_images"   # drop the photos to process here
OUTPUT_DIR = BASE_DIR / "output"        # the program saves finished results here
MASKS_DIR = BASE_DIR / "masks"          # (optionally) drop your own masks here
MODELS_DIR = BASE_DIR / "models"        # AI models (e.g. the depth-estimation model) are cached here, never in AppData/home caches
LIVE_WALLPAPERS_DIR = OUTPUT_DIR / "live_wallpapers"   # live_wallpaper.py saves its exported .mp4 files here
CACHE_DIR = BASE_DIR / "cache"          # resumable per-photo computation cache (depth/segmentation results) - see vfx_engine/cache_store.py

# ---------------------------------------------------------------------------
# Named presets - set to one of
# the names in presets.PRESETS (currently "storm", "gentle_snowfall",
# "misty_dawn") to start this run from that preset's bundle of settings
# instead of the raw individual values below. None (default) = ignore
# presets entirely, use every setting below exactly as written. Applied
# once, at the very start of main()/live_wallpaper.py's entry point -
# every setting the chosen preset mentions OVERRIDES its value below;
# anything the preset doesn't mention keeps its value below unchanged.
# ---------------------------------------------------------------------------
ACTIVE_PRESET = None


def make_sure_folders_exist():
    """Creates the input_images/, output/, masks/, models/, output/live_wallpapers/ and cache/ folders if they don't exist yet."""
    for folder in (INPUT_DIR, OUTPUT_DIR, MASKS_DIR, MODELS_DIR, LIVE_WALLPAPERS_DIR, CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# Different ways of writing the same output mode, so a typo or giving just a
# file extension (e.g. "png" instead of "image") doesn't crash the program.
_OUTPUT_MODE_ALIASES = {
    "image": "image", "img": "image", "static": "image", "photo": "image",
    "png": "image", "jpg": "image", "jpeg": "image", "bmp": "image", "webp": "image",
    "gif": "gif", "animated": "gif", "animation": "gif",
    "mp4": "mp4", "video": "mp4", "mov": "mp4",
}


def resolve_output_mode():
    """Turns OUTPUT_MODE from config.py (e.g. '.png', 'jpg', 'GIF') into one of three modes: 'image' / 'gif' / 'mp4'."""
    key = OUTPUT_MODE.strip().lower().lstrip(".")
    if key not in _OUTPUT_MODE_ALIASES:
        raise ValueError(
            f"Unknown OUTPUT_MODE in config.py: '{OUTPUT_MODE}'. "
            "Use 'image' (or e.g. 'png'/'jpg'), 'gif' or 'mp4' (or 'video')."
        )
    return _OUTPUT_MODE_ALIASES[key]


# ---------------------------------------------------------------------------
# What the program should save for each photo
# ---------------------------------------------------------------------------
# "image" -> a single effect frame, saved as .png (fast, static)
# "gif"   -> an animation saved as .gif
# "mp4"   -> an animation saved as .mp4 (best color quality)
#
# You can also just give the extension, e.g. "png", ".png", "jpg", "gif",
# "mp4", "video" - the program will figure out which of the three modes above
# you mean.
OUTPUT_MODE = "png"

VIDEO_DURATION_SECONDS = 30   # animation length in seconds (only used with OUTPUT_MODE "gif" / "mp4")
FPS = 24                      # frames per second in the gif/mp4

# --- Video encode quality ---
# The old save_mp4() ran with no explicit -crf/-preset at all - meaning it
# silently used ffmpeg's own build defaults (preset "medium", CRF 23), a
# "reasonable web video" setting, not a "quality is the only priority"
# setting. Since generation time isn't a constraint for this project,
# these are pushed hard toward maximum quality per bit instead.
VIDEO_CRF = 14        # lower = higher quality/bigger file; 14 is visually near-lossless for x264
VIDEO_PRESET = "slow"   # slowest/best-compression x264 preset - fine since time isn't a constraint here
# 8-bit (yuv420p) is the deliberate, explicit default for every render mode,
# including live_wallpaper.py - not an oversight. 10-bit (yuv420p10le) would
# reduce banding a bit further in gradient-heavy content, but third-party wallpaper players/decoders (e.g. Wallpaper
# Engine) and some ffmpeg/hardware-decoder builds are 8-bit-only, so 10-bit
# risks a file that won't play at all on the actual target use case. Change
# to "yuv420p10le" only after confirming every intended playback target
# supports it.
VIDEO_PIXEL_FORMAT = "yuv420p"

# Input photos larger than 4K are downscaled before processing (longer side
# scaled to this value, keeping the aspect ratio) - this keeps rendering long
# animations (see VIDEO_DURATION_SECONDS) from using absurd amounts of
# memory/time. Set to None to disable downscaling and always process the
# photo at its original resolution.
MAX_INPUT_DIMENSION = 3840

# Spatial supersampling - the ENTIRE pipeline renders at this many times the final output resolution
# per axis (2 = 4x the pixels), then box-filter-downsamples back down once
# at the very end (see main.py/color.downsample_linear). Rain streaks and
# small snowflakes otherwise only have the shaders' own smoothstep edge-
# softening for anti-aliasing, which visibly aliases/shimmers frame to
# frame at native resolution. 1 disables this (native resolution, the old
# behavior) - meaningfully slower at higher values since every model/
# shader/particle pass now runs at the larger working resolution, but
# generation time isn't a constraint for this project.
SUPERSAMPLE_FACTOR = 1

# Particle size (snowflakes, raindrops) is automatically chosen based on the
# photo's resolution, so particles don't disappear on large photos or look
# oversized on small ones. This multiplier lets you fine-tune that further:
# 1.0 = 100% (automatic size, unchanged), 0.75 = 75% of that size, 1.5 =
# 150%, etc.
PARTICLE_SIZE_SCALE = 1.1

# Particle COUNT also scales automatically with the photo's total pixel
# area (not just particle size, which already scales via
# PARTICLE_SIZE_SCALE above - see _auto_size_scale in particles.py), so
# density-per-visible-area stays roughly consistent across very different
# photo resolutions instead of looking sparser as photos get bigger. Has no
# effect at the 1920x1080 reference resolution the presets in rain.py/
# snow.py were tuned for - it only scales up (or down) from there.
AUTO_DENSITY_SCALE = True

# Cap on how much AUTO_DENSITY_SCALE can multiply particle count by, so a
# very large photo (e.g. 8K) doesn't get an unbounded particle count (and
# render time) - e.g. 4.0 means even a photo with 10x the reference's pixel
# area only ever gets at most 4x the particles.
MAX_DENSITY_SCALE = 4.0

# When True, rain/snow use the photo's real estimated scene depth (see
# vfx_engine/depth.py) to place/size/speed each particle (e.g. rain over
# the near foreground reads as genuinely closer than rain over a distant
# lake) and to blend far particles into the actual background behind them
# (atmospheric perspective - see HAZE_STRENGTH below), instead of a random
# per-particle depth value. Independent of MASK_MODE - this can be on even
# when MASK_MODE isn't "depth". First run downloads/loads the same
# depth-estimation model used by MASK_MODE = "depth" (see config.MODELS_DIR
# - stays inside vfx_overlay/, never a global cache).
PARTICLE_DEPTH_REALISM = True

# 0.0-1.0, how strongly far rain/snow particles blend into the photo's own
# background color behind them when PARTICLE_DEPTH_REALISM is on - real
# atmospheric perspective instead of just shrinking/dimming far particles.
HAZE_STRENGTH = 0.5

# Real camera depth-of-field: None = the old behavior (blur increases as
# particles get farther, near is always sharpest). 0.0-1.0 = a focus
# PLANE at that depth (0.0 = nearest, 1.0 = farthest) - particles away
# from it blur in EITHER direction, matching how a real camera lens
# actually works, instead of "far always blurry, near always sharp."
# 0.5 = focus roughly at mid-distance, a reasonable default for most photos.
FOCUS_DEPTH = 0.5

# 0.0 = off. >0.0, only matters with FOCUS_DEPTH set - particles far from
# the focus plane also grow physically larger (real out-of-focus bokeh
# discs are bigger, not just softer), proportional to how far out of focus
# they are.
BOKEH_FACTOR = 0.6

# rain's depth-of-field above (FOCUS_DEPTH/BOKEH_FACTOR) used
# to be faked entirely via particle size/opacity scaling in
# vfx_engine/particles.py's _setup_particles - out-of-focus drops just got
# bigger and fainter, never an actual blur kernel, which reads as
# recognizably wrong once you know to look for it (no soft spread, no
# disc-shaped highlight - the two hallmarks of real bokeh). Rain now ALSO
# gets a real post-render depth-keyed Gaussian blur pass (see
# vfx_engine/particles_rain.py's _apply_regional_bokeh_blur) layered on
# top of that existing size/opacity scaling, not instead of it - the two
# are complementary (bigger discs from size scaling + an actual blur
# kernel to soften them). RAIN_BOKEH_BLUR_BANDS controls how many discrete
# full-frame blur passes are computed per output frame; results between adjacent bands are
# interpolated per-pixel by defocus amount, so a low band count still
# reads as continuous, not a visibly stepped set of rings. Only affects
# rain (per this item's own scope) - snow's depth-of-field is unchanged.
RAIN_BOKEH_BLUR_BANDS = 4
RAIN_BOKEH_BLUR_MAX_RADIUS_PX = 7.0

# real thin-lens circle-of-confusion (CoC) parameters, replacing
# the old ad hoc `abs(depth - FOCUS_DEPTH) * 2.2` linear defocus curve (which
# lived duplicated in particles.py's _setup_particles and
# particles_rain.py's _apply_regional_bokeh_blur) with the standard optics
# formula CoC = A*f*|S2-S1| / (S2*(S1-f)) - see
# vfx_engine/particles.py's compute_circle_of_confusion for the full
# derivation/citation. This pipeline's depth map is a normalized 0.0-1.0
# value, not a real-world distance, so DOF_NEAR_METERS/DOF_FAR_METERS give
# it a simulated real-world range to be linearly mapped into before the
# formula is evaluated - defaults (1m nearest visible surface, 40m farthest)
# are a reasonable stand-in for this tool's typical outdoor/landscape photo
# subject matter, not a measurement of any specific input photo.
DOF_NEAR_METERS = 1.0
DOF_FAR_METERS = 40.0
# Simulated lens: a 50mm "normal" lens at f/2.8 - a moderate, general-purpose
# depth-of-field look (fast enough to visibly separate focus plane from
# background, not so fast it isolates only a razor-thin sliver in focus).
# Raise DOF_F_NUMBER (e.g. 8-16) for a deep-focus/landscape look, lower it
# (e.g. 1.4-2.0) for a shallow, portrait-like falloff. compute_circle_of_
# confusion self-normalizes its 0..1 output against the worst-case CoC for
# whatever values are set here, so changing these does not require also
# re-tuning FOCUS_DEPTH/BOKEH_FACTOR/RAIN_BOKEH_BLUR_MAX_RADIUS_PX.
DOF_FOCAL_LENGTH_MM = 50.0
DOF_F_NUMBER = 2.8

# how strongly god_rays.sample_light_color's small local
# point-sample (see that function's own docstring) gets blended toward
# lighting.estimate_ambient_light_color's scene-wide ambient term - 0.0
# keeps the old local-sample-only behavior exactly, 1.0 would ignore the
# local sample entirely and use only the whole-photo ambient estimate.
# 0.35 keeps the detected light's own local color dominant (it's still
# the more specific, usually more saturated signal - e.g. an actual warm
# streetlamp) while letting the broader scene cast meaningfully pull it
# toward the photo's overall color temperature instead of one small
# patch's possibly-overexposed/near-white sample dominating completely.
AMBIENT_LIGHT_BLEND_STRENGTH = 0.35

# 0.0-1.0, how much rain/snow particles glow brighter where there's
# something bright directly behind them (a backlit sky, a light source) -
# a rough approximation of real forward light scattering. 0.0 = off.
BACKLIGHT_STRENGTH = 0.7

# 0.0-1.0, symmetric counterpart to BACKLIGHT_STRENGTH above - DARKENS
# rain/snow particles where there's something dark directly behind them
# (deep shade/shadow), instead of only ever brightening them. Fixes the
# "rain/snow brighter than the shadow behind it" bug: without this, a
# backlight-only model left particles rendering at a fixed brightness
# regardless of how dark the background actually was. 0.0 = off.
SHADOW_STRENGTH = 0.6

# Ground-interaction effects (see vfx_engine/ground_fx.py): an animated
# wet-sheen highlight sweep + rain-drop splash rings for rain, a gentle
# gather-and-settle snow build-up for snow. Needs a depth map (the same
# one used by PARTICLE_DEPTH_REALISM/MASK_MODE = "depth") to guess which
# parts of the photo are ground - has no effect without one.
# AI scene segmentation (see vfx_engine/segmentation.py) - classifies WHAT
# each pixel actually is (sky/water/vegetation/ground), not just how far
# it is (that's PARTICLE_DEPTH_REALISM/depth.py above). Currently used to
# sharpen ground_fx's ground-surface targeting below: wet sheen/splashes
# only land on pixels the model actually classifies as paved/bare ground,
# not a lake sitting at "ground-like" depth/position or a raised bench the
# depth-only heuristic alone would mistake for flat ground - see
# ground_fx.build_ground_mask's docstring. Runs once per photo (same
# lazy-load/local-cache pattern as the depth model, cached under
# config.MODELS_DIR - never a global cache). False = ground_fx falls back
# to its pure depth/position heuristic, unchanged.
SEGMENTATION_ENABLED = True

# --- Vegetation sway (see vfx_engine/vegetation.py) - image-space wind
# displacement of the photo's OWN tree/plant/palm/flower/foliage pixels
#, driven by the same curl-noise wind
# field family the particle system uses. Trees/grass
# staying completely static next to falling rain/snow is one of the most
# obvious "this is a static photo with effects on it" tells - motion is
# what human vision is most sensitive to. Only has an effect on
# animations (gif/mp4/live-wallpaper output, i.e. more than one frame) -
# there's nothing to sway in a single static image.
# ---------------------------------------------------------------------------
VEGETATION_SWAY_ENABLED = True
VEGETATION_SWAY_STRENGTH = 0.5    # 0.0-1.0+, overall sway amplitude
VEGETATION_GUST_SCALE = 1.0       # >0, spatial scale of the wind field - smaller = more localized/independent-looking
                                   # gusts across the canopy, larger = broader areas sway together
# Grass/field (segmentation's "ground_soft" category) gets its own
# traveling-wave sway, separate from tree/plant/flower/palm's canopy
# sway -
# real grass has far lower stiffness than a tree canopy and should sway
# MORE readily to the same gust, not less. Multiplies VEGETATION_SWAY_STRENGTH
# (grass reuses the same overall wind-strength knob, just amplified) rather
# than being a wholly separate absolute strength.
GRASS_SWAY_STRENGTH_MULTIPLIER = 1.6

# --- Cloud drift (see vfx_engine/cloud.py) - image-space wind displacement
# of the photo's OWN sky/cloud pixels (from SEGMENTATION_ENABLED's "sky"
# category) - a photo with an otherwise-moving foreground (rain,
# vegetation sway, camera drift) next to a perfectly frozen sky reads as
# visibly wrong once you notice it. Same "animations only" restriction as
# VEGETATION_SWAY_ENABLED above.
# ---------------------------------------------------------------------------
CLOUD_DRIFT_ENABLED = True
CLOUD_DRIFT_STRENGTH = 0.5        # 0.0-1.0+, overall drift amplitude
                                   # direction comes from WEATHER_WIND_DIRECTION_DEG below, not its own setting -
                                   # see the "Weather" section for why

# --- Water ripple -
# image-space wind displacement of the photo's OWN water pixels (from
# SEGMENTATION_ENABLED's "water" category: lake/river/sea/pool). A body
# of water sitting perfectly static for the whole loop while rain falls,
# the camera drifts, and clouds drift directly above it is one of the
# most visually obvious "static photo with effects on it" tells this
# review found - closes it by reusing the same scene_warp.py engine
# vegetation sway and cloud drift are already built on, deliberately
# tied to the SAME wind conditions driving CLOUD_DRIFT_ENABLED above it
# (same noise field timing, same shared weather.gust_envelope) so a
# lake's ripple visibly surges in step with the sky drifting above it,
# not on an independent schedule. Same "animations only" restriction as
# VEGETATION_SWAY_ENABLED/CLOUD_DRIFT_ENABLED above.
# ---------------------------------------------------------------------------
WATER_RIPPLE_ENABLED = True
WATER_RIPPLE_STRENGTH = 0.5       # 0.0-1.0+, overall ripple amplitude - deliberately subtle relative to cloud drift's own scale
                                   # direction comes from WEATHER_WIND_DIRECTION_DEG below, not its own setting - same reason as cloud drift


# ---------------------------------------------------------------------------
# Weather - a single shared wind DIRECTION and GUST timing envelope (see
# vfx_engine/weather.py), sampled by every wind-driven effect below
# (rain/snow particle drift, fog drift, VEGETATION_SWAY_ENABLED,
# CLOUD_DRIFT_ENABLED) instead of each running off its own independent,
# uncoordinated noise field. This is what makes a gust of wind visibly
# bend trees, sway falling snow, and drift fog all in the same direction
# at the same time, instead of several effects that each look fine alone
# but happen to be blowing different directions from each other.
# ---------------------------------------------------------------------------
WEATHER_WIND_DIRECTION_DEG = 15.0   # degrees from horizontal (0 = due right, 90 = straight up) the prevailing wind blows

# the design notes "Fixes to consider" #3: rain's droplet tilt (RAIN_PARAMS used to
# hardcode `angle=18` as an independent fixed value, disconnected from
# WEATHER_WIND_DIRECTION_DEG above) is now computed by main.py's
# resolve_rain_angle_deg() FROM the wind direction every other wind-driven
# effect already shares - see weather.wind_direction_vector(). This is the
# maximum tilt magnitude (degrees from vertical) applied when the wind
# blows due horizontal (the strongest possible horizontal component); it
# scales down toward 0 as WEATHER_WIND_DIRECTION_DEG points more toward
# straight up/down (a headwind/tailwind imparts no sideways tilt on
# falling rain). 18 matches the old fixed storm-rain value, so the current
# WEATHER_WIND_DIRECTION_DEG=15.0 (mostly-horizontal wind) reproduces
# almost the same visual tilt as before, but now genuinely responsive to
# whatever wind direction is actually configured, instead of coincidence.
RAIN_WIND_ANGLE_MAX_DEG = 18.0

# the design notes "Fixes to consider" #3's second half: for single-frame STATIC
# image output (OUTPUT_MODE = "png", num_frames == 1) there is no
# accompanying scene motion at all to back up a strongly wind-tilted rain
# streak - vegetation isn't swaying, clouds aren't drifting, water isn't
# rippling, because none of that has anywhere to animate across in a
# single frame. A full-strength wind tilt on the rain alone, with nothing
# else in frame agreeing it's windy, reads as an isolated/arbitrary slant
# rather than "this scene has wind in it." Multiplies the tilt computed
# from RAIN_WIND_ANGLE_MAX_DEG above DOWN for that case specifically (only
# ever applied when num_frames <= 1 - animated gif/mp4/live-wallpaper
# output is unaffected, since real scene motion IS visible there).
# 1.0 = no reduction, 0.0 = perfectly vertical rain in static images.
RAIN_STATIC_IMAGE_ANGLE_SCALE = 0.4

# the design notes "Fixes to consider" #4's decision: static PNG output (OUTPUT_MODE
# = "image"/"png"/etc, num_frames == 1) architecturally CANNOT show
# wind-driven scene motion (render_vegetation_effects/render_cloud_effects/
# render_water_effects in main.py all early-return [] when num_frames <= 1
# - there's nowhere for a single frame to animate sway/drift/ripple across).
# Rather than either silently accepting that gap forever or forcing
# everyone onto animated output (slower, bigger files, not always wanted),
# the decision made here is: keep static PNG as a supported, first-class
# mode (RAIN_STATIC_IMAGE_ANGLE_SCALE above already keeps its rain tilt
# modest specifically so it doesn't imply motion the image can't back up),
# but have main.py print a one-line advisory recommending OUTPUT_MODE =
# "gif"/"mp4" whenever the CONFIGURED wind is strong enough that even the
# static-scaled-down tilt would still read as "this scene is windy" (see
# main.py's resolve_wind_motion_advisory()). This is deliberately just a
# printed suggestion, not an automatic mode switch or a hard error - which
# output mode to use is the user's call, this only makes the tradeoff
# visible instead of silent. Degrees, measured on the UNSCALED
# (RAIN_WIND_ANGLE_MAX_DEG-based) tilt, i.e. before
# RAIN_STATIC_IMAGE_ANGLE_SCALE is applied - this is about how strong the
# actual configured wind is, not how strong the (deliberately softened)
# rendered rain tilt ends up being.
RAIN_STATIC_WIND_ADVISORY_THRESHOLD_DEG = 8.0

# 0.0 = every effect's motion is numerically IDENTICAL to having no shared
# gust coupling at all (see weather.gust_envelope's docstring) - no
# behavior change, not just a subtle one. >0.0 = how strongly each
# effect's wind-driven motion pulses stronger/calmer together over the loop.
WEATHER_GUST_STRENGTH = 0.35

# a RATE (seconds per full gust pulse), not a fixed total-
# render cycle count - weather._resolve_gust_cycles converts this into a
# whole cycle count (see weather.gust_envelope's docstring for why whole
# numbers only) sized to THIS render's actual duration (VIDEO_DURATION_
# SECONDS/LIVE_WALLPAPER_DURATION_SECONDS), instead of a fixed (2, 3)
# stretched thin (or bunched up) whenever the duration changes. Different
# (coprime-ish) periods for the envelope's two summed sine waves give a
# less mechanically-regular-feeling gust pattern than a single frequency
# would. Tuned so a 30s render reproduces the original (2, 3)-over-30s pacing.
WEATHER_GUST_SECONDS_PER_CYCLE = (15.0, 10.0)

# the design notes ("the gust is a single global scalar - no spatial
# propagation across the frame"): real wind gusts sweep across a
# landscape over real time, not everywhere at once - a value here > 0.0
# makes the gust reach positions further downwind slightly LATER (phase-
# delayed), instead of every pixel/particle feeling it at the exact same
# instant. Expressed as a FRACTION of one gust cycle's period it takes
# for the gust to sweep from one edge of the frame to the opposite edge
# along the wind direction (0.15 = a gust takes 15% of one gust pulse's
# own period to cross the full frame - subtle, since this is meant to be
# a barely-perceptible realism detail on wide landscape shots, not an
# obvious visible wavefront sweeping through). 0.0 = every position feels
# the gust simultaneously - this module's original, pre-13.2 behavior,
# reproduced exactly.
WEATHER_GUST_SPATIAL_DELAY = 0.15


GROUND_FX_ENABLED = True
WET_DARKENING_STRENGTH = 0.4   # 0.0-1.0, how much darker rain-soaked ground gets - this is what
                                # actually reads as "wet" at a glance, more than the highlight below
WET_SHEEN_STRENGTH = 0.4       # 0.0-1.0, brightness of the rain wet-sheen highlight sweep
SPLASH_COUNT = 90              # how many rain splashes appear per loop
SNOW_ACCUMULATION_OPACITY = 0.22   # 0.0-1.0, peak opacity of the gathered-snow overlay

# a RATE (seconds per build-up-and-settle breath), not a
# fixed "exactly one cycle over the entire render" - ground_fx.
# render_snow_accumulation converts this into a whole cycle count sized
# to this render's actual duration, so a long render gets several
# gather/settle breaths instead of one lone cycle stretched across the
# whole thing. Tuned so a 30s render reproduces the original one-breath-
# per-loop pacing.
SNOW_ACCUMULATION_SECONDS_PER_CYCLE = 30.0


# ---------------------------------------------------------------------------
# Mask: which area of the photo to apply the effect to, if not the whole image
# ---------------------------------------------------------------------------
# "none"      -> effect on the whole image, no masking
# "file"      -> mask loaded from the masks/ folder, a file with the same
#                name as the photo (e.g. for input_images/winter1.jpg ->
#                masks/winter1.png). White area of the mask = effect
#                visible, black = invisible.
# "threshold" -> mask computed automatically from the photo's pixel
#                brightness (e.g. "only bright sky")
# "manual"    -> opens a window where you draw the mask with the mouse
#                (left click = draw, right click = eraser, mouse wheel =
#                brush size, Enter = save). The drawn mask is saved to
#                masks/, so the next time the program runs for the same
#                photo the window won't open again - to redraw it, delete
#                the corresponding file from masks/.
MASK_MODE = "depth"

MASK_FEATHER = 15   # blur of the mask edge in pixels, for a soft transition (0 = hard edge)

# "MASK_MODE is a single exclusive switch - masking sources
# can't be combined." ground_fx.build_ground_mask already does exactly
# this kind of combination internally (gating a depth/position heuristic
# by a segmentation mask, multiplicatively) - these two settings expose
# that same capability at the general MASK_MODE level, as OPTIONAL
# secondary layers that further restrict whatever MASK_MODE above
# already produces (either, both, or neither can be active, independent
# of which primary MASK_MODE is chosen - see vfx_engine/mask.py's
# _build_secondary_mask):
#   MASK_SEGMENTATION_CATEGORY: set to a vfx_engine.segmentation category
#       name (e.g. "sky", "water", "vegetation_tree", "ground_hard" - see
#       segmentation.py's _CATEGORY_LABELS for the full list) to ALSO
#       restrict the mask to only that AI-detected region - e.g.
#       MASK_MODE = "depth" + MASK_SEGMENTATION_CATEGORY = "sky" gives
#       "depth occlusion, but only within the detected sky region," one
#       of the design notes own named examples. None (default) = no
#       segmentation-based restriction.
#   MASK_MANUAL_CORRECTION: when True, ALSO restrict the mask by a
#       separate hand-drawn correction layer (masks/{photo}_correction.png,
#       a DIFFERENT file than MASK_MODE = "manual"'s own masks/{photo}.png -
#       drawing/loading works exactly like "manual" mode, just under its
#       own filename) - e.g. MASK_MODE = "depth" + MASK_MANUAL_CORRECTION
#       = True gives "depth occlusion, with a hand-painted correction
#       layered on top to fix a spot the automatic depth mask got
#       wrong," the design notes other named example. False (default) = no
#       manual correction layer.
MASK_SEGMENTATION_CATEGORY = None
MASK_MANUAL_CORRECTION = False

# MASK_MODE = "depth" is the default here so RAIN_USE_MASK/SNOW_USE_MASK
# below (also on by default) actually hide falling rain/snow behind real
# near-foreground objects (a tree, a railing, a rooftop) - the real AI
# scene depth this program already estimates for every photo
# (PARTICLE_DEPTH_REALISM above) was, until this default changed, computed
# but never actually used to occlude particles. See vfx_engine/depth.py
# for the model and vfx_engine/camera.py's apply_camera_drift_to_depth for
# how this stays aligned with CAMERA_DRIFT_ENABLED's zoom/pan.

# only used when MASK_MODE = "threshold"
THRESHOLD_MODE = "bright"   # "bright" = only bright pixels, "dark" = only dark pixels
THRESHOLD_VALUE = 180       # brightness threshold, 0-255

# only used when MASK_MODE = "depth" - see vfx_engine/mask.py's
# _build_depth_mask/_depth_prominence. This is NOT a percentile of raw
# depth ("nearness") - a photo's bottom rows are almost always nearer than
# its top ones from ordinary camera perspective alone (the ground/floor
# recedes into the distance), so thresholding on raw nearness treated that
# entire perspective gradient as one giant "foreground occluder" and hid
# rain/snow across most of the lower frame even where there was no actual
# object there, on top of making the effect fade out right at the point it
# should be visibly reaching the ground. Thresholded on PROMINENCE instead:
# how much NEARER a pixel is than its own broad local surroundings, which
# a real discrete occluder (a tree trunk, a railing, a person, a rooftop
# edge) has and a smoothly receding ground/floor/water plane does not, even
# though the ground plane is still "near" in raw absolute terms. Pixels at
# or above this percentile of prominence (roughly the most-protruding
# `100 - DEPTH_NEAR_PERCENTILE`% of the scene) are treated as foreground
# occluders that the effect falls behind, instead of being painted over them.
DEPTH_NEAR_PERCENTILE = 90

# --- Depth model variant ---
# The depth-estimation model (vfx_engine/depth.py) normally outputs purely
# RELATIVE depth (7.3's logged gap): a vast flat landscape and a tight
# close-up both get their own real depth variation stretched to fill the
# identical 0..1 range, purely as an artifact of per-photo min/max-style
# normalization - so a single fixed DEPTH_NEAR_PERCENTILE/FOCUS_DEPTH value
# behaves inconsistently across different photos for reasons that have
# nothing to do with their actual content. Depth-Anything-V2 also ships
# METRIC checkpoints (fine-tuned on real-world-scale datasets: Hypersim for
# indoor scenes, Virtual KITTI for outdoor ones) that output actual metric
# distance instead - this removes that ambiguity.
#
# "relative" (default) - the original behavior, unchanged.
# "metric"   - loads a metric checkpoint (DEPTH_METRIC_VARIANT below picks
#              indoor vs outdoor) and converts its real-world distance
#              output into this pipeline's existing 0.0 (far) .. 1.0 (near)
#              convention before anything downstream sees it, so every
#              existing depth consumer (occlusion masking, ground_fx,
#              fog falloff, focus-plane blur, camera parallax) keeps
#              working unmodified - only the SOURCE of the depth values
#              changes, not their meaning to callers.
# "marigold" - a completely different ARCHITECTURE (a
#              diffusion model repurposed for depth, via a diffusers
#              pipeline, instead of Depth-Anything's single-pass
#              regression network), aimed specifically at the
#              hazy/low-texture-region weak point named in 18.6 (e.g. bug
#              5.4a's fog-seam issue) - a generative/ensemble-capable model
#              can express "the photo gives almost no signal here" instead
#              of confidently regressing to a plausible-looking wrong
#              value. Same output contract as the other two modes (see
#              vfx_engine/depth.py's estimate_depth docstring for its raw
#              value convention and the flip that normalizes it).
#
# NOTE: the metric checkpoints, and Marigold's checkpoint + its `diffusers`
# package dependency, are NOT part of this project's local install/model
# cache yet (only the relative Small/Large checkpoints under
# models/huggingface, and the transformers/torch already used by every
# other mode, are). depth.py forces fully-offline HuggingFace mode (see its
# own header comment) so nothing is ever silently fetched from the network
# - switching this to "metric" or "marigold" without first installing/
# downloading what each needs (with a real network connection available)
# will raise a clear error telling you so, rather than hanging or silently
# falling back.
DEPTH_MODE = "marigold"

# "outdoor" (Virtual KITTI-trained) or "indoor" (Hypersim-trained) - only
# read when DEPTH_MODE = "metric". This project's typical subject matter
# (ambient/landscape photos - see input_images/) is overwhelmingly outdoor
# scenery, so "outdoor" is the sensible default.
DEPTH_METRIC_VARIANT = "outdoor"


# ---------------------------------------------------------------------------
# monocular surface-normal estimation (a NEW model, not a
# swap of an existing one). Named as "the strongest new-model candidate in
# this document" - closes a gap that's surfaced repeatedly and never had a
# real fix: ground_fx.build_ground_mask's own docstring already admits it
# has no actual surface-flatness/normal detection, which is
# why a splash or wet-sheen highlight can land on a raised, angled surface
# that only *looks* like flat ground from depth+segmentation alone. A
# normal-estimation pass gives real "is this surface actually horizontal
# and flat" data directly, instead of inferring it indirectly through
# stacked depth+position+segmentation heuristics.
#
# Uses diffusers.MarigoldNormalsPipeline (prs-eth/marigold-normals-v1-1) -
# see vfx_engine/surface_normals.py's header comment for the verified API
# details () and this module's output contract (unit-length XYZ vectors in
# [-1, 1], screen-space camera coordinates, passed through unchanged since
# there's no pre-existing consumer contract to match, unlike depth.py's
# multi-mode remapping onto one shared 0.0-far/1.0-near convention).
#
# False (default) - feature is fully implemented but dormant; nothing in
#                    the render pipeline calls estimate_surface_normals()
#                    yet (wiring it into ground_fx is separate follow-on
#                    work, not part of adding the model itself).
# True             - enables vfx_engine.surface_normals.estimate_surface_normals().
#                    Requires the same 'diffusers' package as DEPTH_MODE =
#                    "marigold" above, PLUS the separate
#                    prs-eth/marigold-normals-v1-1 checkpoint (NOT the same
#                    checkpoint as marigold-depth-v1-1, even though both
#                    are Marigold models) - neither is installed/downloaded
#                    in this project yet. Enabling this without first
#                    installing/downloading what it needs (with a real
#                    network connection available) will raise a clear
#                    error telling you so, rather than hanging or silently
#                    falling back, since this module also forces
#                    fully-offline HuggingFace mode.
ENABLE_SURFACE_NORMALS = True


# ---------------------------------------------------------------------------
# learned illumination estimation, via
# diffusers.MarigoldIntrinsicsPipeline's "lighting" variant
# (prs-eth/marigold-iid-lighting-v1-1) - decomposes a photo into Albedo +
# Diffuse shading + Non-diffuse residual (I = Albedo*Shading + Residual).
# See vfx_engine/illumination.py's header comment for the full research
# spike this was built from and the important caveat it documents.
#
# UNLIKE ENABLE_SURFACE_NORMALS above, this item's own the design notes text explicitly
# scopes it as "an exploratory spike ... before committing to a specific
# checkpoint", not a green-lit implementation - the spike's own finding was
# that this checkpoint's diffuse-shading output does NOT cleanly localize a
# discrete light source's screen position/direction/color the way
# god_rays.detect_light_source's brightest-pixel heuristic does, and this
# project's numeric/synthetic-only verification protocol cannot confirm on
# real photos whether it usefully proxies that. It was built anyway, off by
# default, at the user's explicit second confirmation after being shown
# this exact caveat - see the design notes Progress Log entry for
# item 49 for the full reasoning trail.
#
# False (default) - feature is fully implemented but dormant; nothing in
#                    the render pipeline calls estimate_illumination() yet
#                    (wiring its output into god_rays/fog/ground_fx as a
#                    replacement or supplement for sample_light_color is
#                    separate follow-on work, contingent on someone actually
#                    judging its real output against real photos first).
# True             - enables vfx_engine.illumination.estimate_illumination().
#                    Requires the same 'diffusers' package as DEPTH_MODE =
#                    "marigold"/ENABLE_SURFACE_NORMALS above, PLUS the
#                    separate prs-eth/marigold-iid-lighting-v1-1 checkpoint.
#                    Enabling this without first downloading it (with a real
#                    network connection available) will raise a clear error
#                    telling you so, rather than hanging or silently falling
#                    back, since this module also forces fully-offline
#                    HuggingFace mode.
ENABLE_LEARNED_ILLUMINATION = True


# ---------------------------------------------------------------------------
# SAM-based precise mask-boundary segmentation. Described
# in the design notes as "the largest architecture change in this entire
# document", worded repeatedly as conditional on WHETHER to build it at all
# ("only pursue once mask boundary precision specifically is the confirmed
# bottleneck, not preemptively"). Built anyway at the user's own direct,
# repeated request naming SAM segmentation specifically, overriding the
# the design notes own "skip for now" framing - see the design notes
# Progress Log entry for item 47 for the full reasoning trail.
#
# A genuine two-stage pipeline, NOT a swap of segmentation.py's existing
# SegFormer model:
#   Stage 1 (vfx_engine/sam_segmentation.py) - transformers.SamModel +
#   SamProcessor (facebook/sam-vit-huge) propose class-agnostic, pixel-precise
#   mask boundaries via a manually-implemented grid-of-points automatic mask
#   generation loop (transformers doesn't ship the original SAM repo's
#   SamAutomaticMaskGenerator - that lives in the separate 'segment_anything'
#   PyPI package, not a dependency of this project), with stability-score
#   filtering and greedy IoU-based deduplication of overlapping proposals.
#   SAM itself produces no semantic labels at all.
#   Stage 2 - each SAM-proposed region is assigned one of segmentation.py's
#   9 existing category labels by reusing estimate_segmentation(image)'s
#   per-pixel probability arrays directly (mean probability per category
#   over the region's pixels, argmax above a confidence threshold), instead
#   of a separate CLIP-based zero-shot classifier.
#
# False (default) - feature is fully implemented but dormant; nothing in the
#                    render pipeline calls estimate_sam_segmentation() yet
#                    (wiring its precise per-category masks in as a
#                    replacement/refinement for segmentation.py's soft
#                    probability masks in ground_fx or elsewhere is separate
#                    follow-on work, not part of adding the model itself).
# True             - enables vfx_engine.sam_segmentation.estimate_sam_segmentation().
#                    Requires the same 'transformers' package already used by
#                    SEGMENTATION_ENABLED above (facebook/sam-vit-huge is
#                    loaded via transformers.SamModel/SamProcessor - no new
#                    PyPI dependency), PLUS the separate facebook/sam-vit-huge
#                    checkpoint (~2.4GB, NOT downloaded in this project yet).
#                    Enabling this without first downloading it (with a real
#                    network connection available) will raise a clear error
#                    telling you so, rather than hanging or silently falling
#                    back, since this module also forces fully-offline
#                    HuggingFace mode.
ENABLE_SAM_SEGMENTATION = False


# ---------------------------------------------------------------------------
# Camera - a slow, cyclic zoom/pan (Ken Burns effect) applied to the base
# photo itself before effects are composited on top. See vfx_engine/camera.py.
# Only used when num_frames > 1 (i.e. gif/mp4/live-wallpaper output).
# ---------------------------------------------------------------------------
CAMERA_DRIFT_ENABLED = True

# Real depth-driven multi-layer parallax - splits the photo
# into near/mid/far depth layers (vfx_engine.camera.apply_layered_camera_drift)
# and lets each drift at its own effective amplitude, instead of one flat
# image warped by a single rigid transform (which moves a near tree and a
# distant mountain by the identical pixel amount - the single highest-
# impact "does this feel like a real filmed camera move" gap this pass
# found). Falls back to the old flat drift automatically when no depth
# map is available (e.g. MASK_MODE != "depth" and PARTICLE_DEPTH_REALISM
# off - see resolve_particle_depth_realism/MASK_MODE above).
CAMERA_PARALLAX_ENABLED = False

CAMERA_ZOOM_AMPLITUDE = 0.05    # 0.0-1.0, how far the "camera" zooms in at the peak of the loop
CAMERA_PAN_AMPLITUDE = 0.045    # 0.0-1.0, how far it drifts sideways/vertically at the peak, as a fraction of width/height
CAMERA_PAN_PHASE_DEG = 100.0    # phase offset between horizontal/vertical pan - 90-180 gives a real looping ellipse, not a back-and-forth line

# a RATE (seconds per zoom/pan cycle), not a fixed "exactly
# one cycle over the entire render" - vfx_engine.camera._camera_path
# converts this into a whole cycle count sized to this render's actual
# duration, so a long render gets several drift cycles instead of one
# lone zoom/pan stretched across the whole thing (which reads as an
# almost-static camera on anything past ~30-60s). Tuned so a 30s render
# reproduces the original one-cycle-per-loop pacing.
CAMERA_DRIFT_SECONDS_PER_CYCLE = 30.0

# Alternate zoom envelope shape - see vfx_engine.camera._camera_path:
#   "breathe"     - the original pattern: 1.0 -> 1.0+CAMERA_ZOOM_AMPLITUDE
#                   -> 1.0, eased with a raised cosine.
#   "breathe_out" - the same shape, phase-inverted: starts/ends at the
#                   PEAK zoom and eases OUT to 1.0 at the loop's midpoint.
#   "ease_hold"   - the same envelope, reshaped via smoothstep for more
#                   dwell time at the loop's rest/peak framing and a
#                   snappier sweep through the middle.
CAMERA_ZOOM_PATTERN = "breathe"

# Composition-aware pan direction: when enabled, the pan ellipse's
# dominant drift direction is rotated to lean toward the photo's own
# detected dominant light source (see god_rays.detect_light_source, the
# same detection god rays/lens flare/rain's dual-tone lighting already
# reuse) instead of always tracing the same content-agnostic diagonal
# regardless of what's actually in the frame - see
# main.py's resolve_camera_composition_kwargs and
# vfx_engine.camera._camera_path's pan_bias_angle_deg.
CAMERA_COMPOSITION_AWARE_ENABLED = True


# ---------------------------------------------------------------------------
# the design notes "Fixes to consider" #7 - base-photo color-grade unification pass
# (see vfx_engine/color_grade.py). This pipeline has always only ADDED
# effect layers on top of the base photo - nothing has ever corrected the
# base photo's OWN lighting. A source photo with a warm sunset sky over a
# cool-toned foreground (or the reverse) stayed exactly that inconsistent,
# and every effect that samples the photo's own color for its own tint
# (rain/splash/lens-flare highlight color, etc.) inherited whichever
# mismatch was already baked in. Applied once, early in main.py's
# process_image - right after segmentation resolves, before ANY effect
# samples the photo - so every downstream consumer automatically inherits
# the corrected, unified grade with no changes needed anywhere else, same
# "fix it once, upstream" precedent as SUPERSAMPLE_FACTOR/CAMERA_DRIFT above.
# Deliberately gentle/partial (COLOR_GRADE_UNIFY_STRENGTH below) - a photo
# already lit coherently comes out of this pass visually unchanged; see the
# module docstring for why.
# ---------------------------------------------------------------------------
# Weather-specific grade of the base photo (vfx_engine/weather_grade.py):
# None, "sunny" (dehaze/warm/contrast/blue sky) or "storm" (dark/cool/desaturated).
WEATHER_GRADE_STYLE = None
WEATHER_GRADE_STRENGTH = 1.0
COLOR_GRADE_UNIFY_ENABLED = True
COLOR_GRADE_UNIFY_STRENGTH = 0.5   # 0.0 (no correction) - 1.0 (fully match the sky/ground color casts)


# ---------------------------------------------------------------------------
# Lighting - volumetric light shafts ("god rays"), a simple camera lens
# flare, and a slow ambient color drift, all derived from the photo's own
# detected dominant light source. See vfx_engine/god_rays.py.
# ---------------------------------------------------------------------------
GOD_RAYS_ENABLED = True
GOD_RAYS_THRESHOLD = 0.55    # 0.0-1.0, how bright the (already radially-blurred) result must be to show as a ray
GOD_RAYS_STRENGTH = 0.7      # 0.0-1.0+, overall ray brightness
GOD_RAYS_COLOR = (255, 244, 214)   # warm light color
GOD_RAYS_PULSE_AMOUNT = 0.2  # 0.0-1.0, how much the rays waver in intensity over the loop (real light through moving haze isn't static)

# a photo can have several comparably bright light sources
# (streetlights down a road, multiple windows/headlights), not just one
# dominant sun/lamp - god_rays.detect_light_sources returns up to
# MAX_LIGHT_SOURCES of them (dominant first, relative strength 1.0), and
# god rays / lens flare / starburst each render one pass per detected
# source, scaled down by that source's relative strength so secondary
# lights read as genuinely fainter, not identical copies of the main one.
MAX_LIGHT_SOURCES = 3
MIN_LIGHT_SOURCE_RELATIVE_STRENGTH = 0.35   # a secondary source dimmer than this fraction of the dominant one is ignored

LENS_FLARE_ENABLED = False  # off by default: detect_light_sources fires on water/reflection highlights, not just real light sources, and the ghost chain's chromatic-dispersion jitter (_jitter_ghost_color) can land visibly red/orange ghosts scattered across the whole frame
LENS_FLARE_STRENGTH = 0.4
LENS_FLARE_COLOR = (255, 250, 235)
LENS_FLARE_GHOST_COUNT = 3

# Star-shaped diffraction spikes at the detected light source - a new capability, not present before; see
# vfx_engine/god_rays.py's build_starburst_texture.
STARBURST_ENABLED = False  # off by default: same detect_light_sources false-positive issue as LENS_FLARE_ENABLED above - fires on water reflections, not just real light sources
STARBURST_STRENGTH = 0.55
STARBURST_COLOR = (255, 252, 240)
STARBURST_SPIKE_COUNT = 6          # spike pairs (6 -> a 12-point star, the classic sharp-aperture look)
STARBURST_LENGTH_FRACTION = 0.42   # spike length as a fraction of min(width, height)
STARBURST_ANGULAR_SHARPNESS = 48.0 # higher = thinner, sharper spikes
STARBURST_PULSE_AMOUNT = 0.12      # 0.0-1.0, subtle intensity waver over the loop

# Slow, cyclic (loop-safe) ambient color drift over the loop - real
# ambient light isn't perfectly constant even over a short span (drifting
# clouds, atmosphere) - see vfx_engine/lighting.py's apply_ambient_drift.
AMBIENT_DRIFT_ENABLED = True
AMBIENT_DRIFT_STRENGTH = 0.06   # 0.0-1.0, how far the tint swings warm/cool at the peaks of the cycle

# a RATE (seconds per warm<->cool swing), not a fixed
# "exactly one swing over the entire render" - vfx_engine.lighting.
# apply_ambient_drift converts this into a whole cycle count sized to
# this render's actual duration, so a long render gets several swings
# instead of one lone swing stretched across the whole thing. Tuned so a
# 30s render reproduces the original one-swing-per-loop pacing.
AMBIENT_DRIFT_SECONDS_PER_CYCLE = 30.0

# Camera-lens grime/dust (see vfx_engine/lens_dirt.py) - a static, subtle
# darkening layered on top of everything else (a camera artifact, like
# lens flare, not scene content).
LENS_DIRT_ENABLED = True
LENS_DIRT_STRENGTH = 0.3   # 0.0-1.0, how visible the grime is

# grime near the photo's own detected light source scatters/
# glows toward that light's color instead of only ever darkening - see
# vfx_engine/lens_dirt.py's render_lens_dirt.
LENS_DIRT_GLOW_STRENGTH = 0.9   # 0.0-1.0+, how strongly the glow can overpower plain darkening at its brightest
LENS_DIRT_GLOW_RADIUS = 0.55    # >0, aspect-corrected falloff radius - matches fog's own light_radius (shaders/blur.glsl)

# real dust/grime is far more visible against a bright, even
# background (open sky) than a dark one (shadowed foliage) - grime alpha
# is now modulated by the photo's own local brightness underneath instead
# of applying uniformly regardless of what's behind it.
LENS_DIRT_BRIGHTNESS_MIN_VISIBILITY = 0.35   # >=0, alpha multiplier over the darkest background regions
LENS_DIRT_BRIGHTNESS_MAX_VISIBILITY = 1.35   # >=min, alpha multiplier over the brightest background regions (can exceed 1.0 - final alpha is still clipped)
LENS_DIRT_BRIGHTNESS_BLUR_FRACTION = 0.03    # >0, Gaussian blur radius for the brightness reference, as a fraction of image width - smooths pixel-level noise into a "local background" read

# real lens imperfections are usually a mix of a broad, soft
# overall haze film AND a handful of small, sharp, sparse dust specks/
# hairs - one homogeneous noise field can only convincingly read as one of
# the two. build_dirt_texture now combines two independently-seeded fbm
# layers at different frequencies/thresholds instead of one.
LENS_DIRT_SPECK_SCALE = 14.0       # spatial frequency of the fine dust-speck layer - much higher than the haze film's default scale=3.0
LENS_DIRT_SPECK_THRESHOLD = 0.78   # 0.0-1.0, higher than the haze film's default threshold=0.6 - keeps specks sparse and distinct rather than another broad film
LENS_DIRT_SPECK_EDGE_POWER = 0.6   # >0, much gentler edge bias than the haze film's default edge_power=2.4 - specks (dust motes, hairs) can land anywhere on a lens, not just the edges
LENS_DIRT_SPECK_WEIGHT = 0.7       # >=0, how strongly the speck layer contributes once combined with the haze film

# the design notes "Fixes to consider" #6: soft, out-of-focus water droplets sitting
# directly on the camera lens itself (see vfx_engine/lens_rain.py) - a
# camera artifact distinct from LENS_DIRT_* above (static dust/grime film)
# and distinct from RAIN_PARAMS below (sharp in-scene falling drops).
# Gated to heavy-rain intensity specifically - see main.py's
# render_lens_rain_effects, which only enables this when RAIN_ENABLED and
# RAIN_PARAMS["intensity"] == "storm": light/medium rain doesn't throw
# enough water at an exposed lens for this to be realistic.
LENS_RAIN_ENABLED = True
LENS_RAIN_REQUIRED_INTENSITY = "storm"   # RAIN_PARAMS["intensity"] value that unlocks this effect
LENS_RAIN_DROPLET_COUNT = 6             # how many droplets sit on the lens
LENS_RAIN_RADIUS_RANGE = (6, 18)        # (min, max) droplet minor-axis radius in pixels
LENS_RAIN_ELONGATION_RANGE = (1.05, 1.55)   # (min, max) gravity-stretched teardrop major/minor ratio
LENS_RAIN_MAGNIFICATION = 0.55           # 0.0-1.0+, how strongly each droplet bulges/compresses its refracted background
LENS_RAIN_BLUR_RADIUS_FRACTION = 0.012   # >0, out-of-focus Gaussian blur radius as a fraction of image width
LENS_RAIN_STRENGTH = 0.85                # 0.0-1.0+, overall droplet opacity
LENS_RAIN_RIM_DARKEN_STRENGTH = 0.25     # 0.0-1.0+, how strongly the meniscus edge ring darkens each droplet
LENS_RAIN_WOBBLE_PX = 1.5                # >=0, amplitude (px) of each droplet's closed-loop surface-tension jiggle
LENS_RAIN_PULSE_AMOUNT = 0.25            # 0.0-1.0+, how much the specular highlight brightens/dims over the loop
LENS_RAIN_PULSE_SPEED = 1                # whole number of pulse cycles per loop - MUST stay whole so the loop closes exactly
LENS_RAIN_HIGHLIGHT_STRENGTH = 0.8       # 0.0-1.0+, brightness of each droplet's specular glint


# ---------------------------------------------------------------------------
# Effects - enable/disable and configure each one
# ---------------------------------------------------------------------------

# --- Snow ---
SNOW_ENABLED = False
SNOW_USE_MASK = True   # True = only within the mask, False = the whole image - with MASK_MODE = "depth" (the
                        # default above) this is what makes snow disappear behind near-foreground objects
                        # instead of floating flatly in front of literally everything in the photo
SNOW_PARAMS = dict(
    density=0.6,   # 0.0-1.0, how many flakes are on screen at once
    wind=0.4,      # 0.0-1.0, how strongly flakes sway sideways
    depth=True,    # vary size/speed based on simulated distance
    # Real sub-frame motion blur - snow falls much slower
    # than storm rain, so this matters less here, but it's free
    # consistency to leave it on; 2 samples is enough for slow-moving
    # flakes (see RAIN_PARAMS below for the fast-motion case).
    motion_blur_samples=2,
    shutter_fraction=0.5,
)

# --- Rain ---
RAIN_ENABLED = True
RAIN_USE_MASK = True   # see SNOW_USE_MASK above - same depth-occlusion behavior, for rain
RAIN_PARAMS = dict(
    intensity="storm",    # "light" / "medium" / "storm"
    # angle is intentionally NOT set here:
    # a fixed independent tilt value could point rain any direction
    # regardless of which way config.WEATHER_WIND_DIRECTION_DEG says the
    # scene's wind is actually blowing - main.py's resolve_rain_angle_deg()
    # computes it instead, from the SAME shared wind direction every other
    # wind-driven effect (vegetation sway, cloud drift, water ripple) uses,
    # so a wind-tilted rain streak actually agrees with which way the rest
    # of the scene's wind is blowing. See RAIN_WIND_ANGLE_MAX_DEG/
    # RAIN_STATIC_IMAGE_ANGLE_SCALE below for the two knobs that shape it.
    blur=True,             # motion-blur streaks behind the drops
    # Real sub-frame motion blur, replacing the old
    # postprocess.py whole-frame ghosting hack for exactly the case that
    # write-up calls out as mattering most ("especially on fast rain
    # streaks") - 3 samples across a 180-degree-shutter-equivalent window.
    motion_blur_samples=3,
    shutter_fraction=0.5,
    # User-requested global fall-speed knob (2026-09-22): multiplies every
    # intensity preset's speed_range (see _INTENSITY_PRESETS in
    # vfx_engine/particles_rain.py) uniformly, independent of which
    # intensity is chosen and independent of RAIN_NEAR_LAYER/RAIN_FAR_LAYER's
    # own speed_scale (which further scales near vs far relative to each
    # other, on top of this). 1.0 = old behavior, unscaled. Threads through
    # automatically to both the single-pass and layered-rain code paths
    # (see main.render_enabled_effects and main.render_layered_rain's
    # `shared` dict) since render_rain() now accepts this same key.
    speed_scale=1.0,
)

# split rain into two independently-tuned near/far depth
# layers (a big/blurry/sparse near pass and a small/sharp/numerous far
# pass) instead of one continuous depth gradient - a single gradient can
# already size/speed/blur drops continuously by real scene depth (see
# PARTICLE_DEPTH_REALISM), but can't give the two ends of that range their
# own streak length or wind angle the way two visually distinct layers of
# real rain at different distances actually look. This renders rain
# TWICE per frame, so it's gated behind its own flag rather than
# always-on (see main.render_layered_rain) - only takes effect when
# PARTICLE_DEPTH_REALISM is also on and a depth map was actually computed;
# falls back to the old single-pass rain automatically otherwise.
RAIN_DEPTH_LAYERS_ENABLED = True
RAIN_DEPTH_OCCLUSION_STRENGTH = 0.75   # 0.0-1.0, how strongly drops behind a nearer scene surface (e.g. a foreground tree) are hidden
RAIN_NEAR_LAYER = dict(
    count_scale=0.4,            # near drops are few but big/prominent
    size_scale=1.5,
    speed_scale=0.85,           # screen-space: bigger near drops read as slower relative to frame despite covering more real distance
    velocity_stretch_scale=1.6,  # longer motion-blur streak seed
    depth_bias_power=0.5,       # skews sampled depth UP toward 1.0 (near)
    angle_offset_deg=-3.0,      # slightly different wind read at close range
)
RAIN_FAR_LAYER = dict(
    count_scale=1.2,            # far drops are numerous but small/sharp
    size_scale=0.85,
    speed_scale=1.1,
    velocity_stretch_scale=0.6,
    depth_bias_power=2.2,       # skews sampled depth DOWN toward 0.0 (far)
    angle_offset_deg=3.0,
)

# --- Fog / smoke / steam ---
# the design notes "Fixes to consider" #5: turned on so the pipeline's own
# depth-aware, depth-smoothed fog (FOG_DEPTH_AWARE/FOG_DEPTH_SMOOTHING_PX
# below) actively masks over and unifies whatever mist/haze is already
# baked into a source photo instead of leaving that pre-existing, pipeline-untouched haze
# sitting there inconsistently. FOG_DEPTH_AWARE=True means density still
# scales with each photo's own real depth - a photo with little natural
# haze (e.g. castle_by_the_lake.png) gets a much thinner, subtler layer
# than a genuinely misty one, so turning this on globally doesn't plaster
# uniform fog onto every photo regardless of how much it needs it.
FOG_ENABLED = False
FOG_USE_MASK = False
FOG_PARAMS = dict(
    density=0.28,
    movement=0.35,   # radius of the (closed) drift path over the loop - see vfx_engine/fog.py
    color=(215, 220, 230),
)
# When True (and a depth map is available - see PARTICLE_DEPTH_REALISM/
# MASK_MODE), fog density scales with real scene depth - genuinely
# thicker in the distance, thinner in the near foreground - instead of a
# flat density plastered evenly over the whole frame. See vfx_engine/fog.py.
FOG_DEPTH_AWARE = True
FOG_DEPTH_POWER = 1.6   # >0, higher keeps the near foreground clearer for longer before fog ramps up
# Gaussian blur radius (px) applied to the depth-to-density gradient before
# it's used as fog alpha - fixes a confirmed bug: monocular
# depth models are weak on large, low-texture, hazy regions (a distant
# misty forest canopy is a textbook case) and can introduce a discontinuous
# depth jump right through such a region with nothing else in the pipeline
# to catch it, which without this smoothing shows up as a hard visible
# line cutting through the fog layer. Fog is meant to read as soft
# atmospheric haze anyway, so blurring its density field costs nothing
# visually. 0 disables the smoothing.
FOG_DEPTH_SMOOTHING_PX = 25
# when True, fog also picks up a lightly-blurred grayscale of
# the base photo itself, brightening slightly over already-bright ground
# and thinning along the photo's own strong edges (so it pools around
# rocks/fences/treelines instead of smearing across them), instead of
# depending only on procedural noise/depth/vertical position with zero
# relationship to actual scene content. See vfx_engine/fog.py.
FOG_GROUND_TEXTURE_ENABLED = True
FOG_GROUND_TEXTURE_STRENGTH = 0.3   # 0.0-1.0, how strongly fog anchors to the photo's own ground detail

# --- Light flicker (e.g. fire, lamp) ---
FLICKER_ENABLED = False
FLICKER_USE_MASK = False   # True = only flickers within the mask (e.g. only around a lamp) - this is the GENERIC depth/segmentation shared_mask (mask.py), with no inherent relationship to where the light actually is; see FLICKER_AUTO_LIGHT_MASK below for that
# when True, flicker is scoped to a soft Gaussian
# falloff mask centered on the photo's own AUTO-DETECTED dominant light
# position (god_rays.detect_light_source, via lighting.build_light_position_mask
# - the identical falloff blur.glsl's light_glow term already uses),
# instead of (or as well as - this takes priority when both are on)
# FLICKER_USE_MASK's light-unaware shared_mask. Off by default: an
# incorrectly-applied flicker on a photo whose "dominant light" is
# actually the sun or a bright sky, not a lamp/campfire, would look wrong
# - turn this on for photos where the detected light really is the
# flickering source itself.
FLICKER_AUTO_LIGHT_MASK = False
FLICKER_LIGHT_MASK_RADIUS = 0.55   # matches blur.glsl's own default light_radius, in the same aspect-corrected UV units
FLICKER_PARAMS = dict(
    intensity=0.15,             # 0.0-1.0, maximum brightening strength
    speed=0.3,                  # how fast it flickers
    color=(255, 220, 150),      # glow color (warm light)
)

# --- Lightning (procedural bolt + full-frame flash) ---
LIGHTNING_ENABLED = False
LIGHTNING_PARAMS = dict(
    frequency=0.15,              # roughly how many strikes per second on average (e.g. 0.15 =~ one every ~7s at 24 FPS)
    intensity=0.9,                # 0.0-1.0, brightness of the flash
    color=(230, 235, 255),        # bolt/flash color (cool white-blue)
    branch_chance=0.35,           # 0.0-1.0, how likely each bolt segment is to spawn a side branch
    strike_frames=3,              # how many frames one strike lasts (bright flash, then a couple of dimmer frames)
    # Flash spatial falloff - the flash is brightest at the
    # bolt's own strike point and fades toward (not to zero, real flash
    # still lights the whole sky, just unevenly) min_intensity_frac by
    # falloff_radius_frac of the frame diagonal away. See lightning.py's
    # apply_flash/_radial_flash_mask.
    flash_falloff_radius_frac=0.85,
    flash_min_intensity_frac=0.25,
)


# ---------------------------------------------------------------------------
# Post-processing - a screen-space pass applied to the fully finished frames,
# right before saving. See vfx_engine/postprocess.py. Set any stage to None
# to skip it entirely.
#
# bloom/tonemap turned on by default: the cheapest available
# quality win - both were already fully implemented but switched off,
# meaning nothing on any render actually glowed or rolled off softly.
# chromatic_aberration is left off for now - it has its own falloff-curve/
# subpixel fixes still pending and isn't part
# of this "free win" pass. motion_blur is left off by default; turn it on here (see
# vfx_engine/postprocess.py's _apply_motion_blur and main.py's
# resolve_postprocess_params, which supplies the per-frame "vectors" this
# stage needs from the render's own actual camera path) if a more
# cinematic streak on the camera pan/zoom itself is wanted. grain
# is on by default - a subtle, animated fine-noise layer,
# since a mathematically noise-free composite is one of the most
# recognizable "rendered, not photographed" tells; its noise is scaled
# per-pixel by luminance, so
# shadows read grainier than highlights, the way real sensor/film noise
# actually behaves.
# ---------------------------------------------------------------------------
POSTPROCESS_PARAMS = dict(
    bloom=dict(threshold=0.75, strength=0.35),   # gentle glow on bright areas (god rays, wet sheen, lightning)
    chromatic_aberration=None,    # e.g. dict(strength=2.0) - subtle lens-like red/blue channel shift near the edges
    motion_blur=None,             # e.g. dict(strength=0.35) - directional streak along the camera's own drift path (see above); "vectors" is filled in automatically by main.py/live_wallpaper.py, do not set it here
    tonemap=dict(strength=0.35),  # subtle filmic highlight roll-off, softens instead of clipping to flat white
    grain=dict(strength=0.018),   # subtle animated film grain (luminance-scaled - see above) - also doubles as dither ahead of the final 8-bit encode
)


# ---------------------------------------------------------------------------
# Live wallpapers - see live_wallpaper.py. Exports a short, seamlessly
# looping .mp4 to output/live_wallpapers/, meant to be set as an animated
# desktop background with third-party tools (e.g. Wallpaper Engine) that
# accept a video file - this script does NOT touch your actual Windows
# desktop background itself.
# ---------------------------------------------------------------------------
LIVE_WALLPAPER_DURATION_SECONDS = 30   # length of the exported loop
LIVE_WALLPAPER_CROSSFADE_SECONDS = 1.5   # how much of the tail is blended into the head to hide the loop seam - see vfx_engine/loop.py
# CRF-driven: the old fixed 8Mbps
# target was a modest cap for what can be a 4K 30-second loop. Since the
# expensive GPU render is only ever consumed once (into a lossless FFV1
# intermediate - see save_mp4_two_pass), there's no reason not to let the
# final encode use as many bits as CRF 14/veryslow actually wants.
LIVE_WALLPAPER_CRF = 14

# Longest side (px) of the live-wallpaper video. None = the photo's own size.
# A smaller value renders proportionally faster (the GUI's Draft / Balanced /
# High quality choices set this); photos already smaller than the cap are
# never scaled up.
LIVE_WALLPAPER_MAX_DIMENSION = None

# True = also open a window that plays the exported loop live, right after
# exporting it, so you can preview it without leaving the program.
LIVE_PREVIEW_WINDOW = True
