# Marks vfx_engine/ as a Python package, so it can be imported as
# `from vfx_engine import ...` from anywhere in the project.
#
# Modules in this folder:
#   gpu.py         - shared GPU context (moderngl) and shader loading
#   particles.py   - particle engine (snow, rain)
#   snow.py        - snow effect
#   rain.py        - rain effect
#   fog.py         - fog/smoke/steam effect
#   lighting.py    - light flicker
#   mask.py        - masks (which area of the image the effect is applied to)
#   mask_editor.py - window for manually drawing a mask with the mouse
#   compositor.py  - combining effects and compositing them onto the image
#   image_io.py    - loading images and saving the result
