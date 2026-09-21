"""
gpu.py

Small shared helpers for talking to the GPU through moderngl:
    - one OpenGL context, reused by every effect instead of creating a new
      one each time (context creation is the slow part)
    - a helper that loads one of our shaders/*.glsl files and compiles it
      into a moderngl program
    - a helper that loads one of our shaders/*.comp compute shaders
    - capability detection for compute shaders (OpenGL 4.3+)

We use `moderngl` instead of raw PyOpenGL - it's the exact same real GPU
shaders (GLSL) underneath, just without the large amount of manual
boilerplate (window/context creation, buffer management) that raw PyOpenGL
needs.
"""

import moderngl

import config

_context = None  # created once, on first use
_has_compute = None  # cached result of supports_compute()

_COMPUTE_MIN_VERSION = 430  # OpenGL 4.3 is when compute shaders were added to core


def get_context():
    """
    Returns the shared, "headless" (no visible window) OpenGL context used
    by every GPU effect. Created once per program run.

    moderngl's default standalone context request often negotiates down to
    an old baseline version (3.3) even on GPUs that support much more -
    so we explicitly ask for 4.3 (needed for compute shaders, see
    supports_compute()) and gracefully fall back to whatever the driver
    gives us by default if that request fails (e.g. very old integrated
    GPUs, some virtual machines).
    """
    global _context
    if _context is None:
        try:
            _context = moderngl.create_context(standalone=True, require=_COMPUTE_MIN_VERSION)
        except Exception:
            _context = moderngl.create_context(standalone=True)
    return _context


def supports_compute():
    """
    Returns True if the current GPU/driver can run compute shaders (OpenGL
    4.3+). Effects that have a compute-shader fast path (see particles.py)
    check this once and fall back to the equivalent CPU (numpy) computation
    when it's False, so the program still runs correctly on older hardware.
    """
    global _has_compute
    if _has_compute is None:
        _has_compute = get_context().version_code >= _COMPUTE_MIN_VERSION
    return _has_compute


def load_shader_program(filename):
    """
    Loads one of our shaders/*.glsl files (see shaders/particles.glsl for
    the format) and compiles it into a ready-to-use moderngl program.

    Our .glsl files hold BOTH the vertex and fragment shader in one file,
    split by "#vertex" / "#fragment" marker lines - not a real GLSL feature,
    just a simple convention so related shader code stays in one place.
    """
    path = config.BASE_DIR / "vfx_engine" / "shaders" / filename
    vertex_source, fragment_source = split_shader_sections(path.read_text(encoding="utf-8"))

    context = get_context()
    return context.program(vertex_shader=vertex_source, fragment_shader=fragment_source)


def load_compute_program(filename):
    """Loads one of our shaders/*.comp compute shaders and compiles it into a ready-to-use moderngl compute program."""
    path = config.BASE_DIR / "vfx_engine" / "shaders" / filename
    source = path.read_text(encoding="utf-8")

    context = get_context()
    return context.compute_shader(source)


def split_shader_sections(source):
    """
    Splits one of our .glsl files into its vertex/fragment halves. The
    marker has to be a WHOLE line by itself ("#vertex" / "#fragment", no
    other text on that line) - a naive substring search would also match
    the words "#vertex" / "#fragment" if they show up in a comment
    somewhere (like the explanation at the top of these files does).
    """
    section = None
    sections = {"vertex": [], "fragment": []}

    for line in source.splitlines():
        stripped = line.strip()
        if stripped == "#vertex":
            section = "vertex"
            continue
        if stripped == "#fragment":
            section = "fragment"
            continue
        if section is not None:
            sections[section].append(line)

    return "\n".join(sections["vertex"]), "\n".join(sections["fragment"])
