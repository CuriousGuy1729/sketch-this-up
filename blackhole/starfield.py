"""
Stage 4 -- the sky: a procedural star field plus a NumPy-synthesised nebula.

Two complementary layers:

1. **Stars -- procedural (Voronoi).**  Deliberately *not* a texture.  Near the
   photon ring the lensing magnification diverges, so any finite-resolution
   environment map turns into visible mush exactly where the image is most
   interesting.  A procedural field stays sharp under arbitrary magnification,
   which is what makes the Einstein rings read as rings of individual stars.
   Three octaves give a plausible magnitude distribution, and each star is
   tinted by a blackbody drawn from a realistic stellar temperature range.

2. **Nebula / Milky Way -- a NumPy equirectangular map.**  Diffuse, so it does
   not need to survive magnification.  Generated with spectrally-shaped
   (1/f^beta) fractal noise, which is a far better model of interstellar
   structure than layered Perlin, plus a galactic band and dust extinction.

Because the sky is sampled with the *bent* ray direction after the portal
shells are done with it, all of this is lensed automatically.
"""

from __future__ import annotations

import os

import bpy
import numpy as np

from .config import Config
from .nodes import Expr, NodeBuilder


# ---------------------------------------------------------------------------
# NumPy nebula synthesis
# ---------------------------------------------------------------------------

def _fractal_field(shape: tuple[int, int], beta: float, rng: np.random.Generator
                   ) -> np.ndarray:
    """
    Fractal noise with a 1/f^beta power spectrum, built in Fourier space.

    Shaping the spectrum directly (rather than summing octaves of Perlin) is
    both faster at these resolutions and a better match to the observed
    power-law structure of interstellar gas.
    """
    h, w = shape
    white = rng.normal(size=(h, w))
    spectrum = np.fft.rfft2(white)

    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    k[0, 0] = 1.0
    spectrum *= k ** (-beta / 2.0)
    spectrum[0, 0] = 0.0

    field = np.fft.irfft2(spectrum, s=(h, w))
    # Standardise to z-scores rather than rescaling min->max.  A 1/f^beta
    # field has unbounded tails, so min/max normalisation makes the usable
    # range depend on whatever single extreme texel the RNG happened to
    # produce -- and then a threshold like "> 0.58" covers wildly different
    # fractions of the sky run to run.  With z-scores a threshold means a
    # fixed sky coverage via the normal CDF, which is what lets the sky below
    # stay mostly black.
    field -= field.mean()
    sigma = field.std()
    return field / sigma if sigma > 0 else field


def generate_nebula_image(cfg: Config) -> bpy.types.Image:
    """
    Synthesise an equirectangular deep-sky background and hand it to Blender
    as a float image (never written to disk, so nothing to clean up).
    """
    width = int(cfg.stars.nebula_width)
    height = width // 2
    rng = np.random.default_rng(cfg.stars.seed)

    # Spherical coordinates of every texel.
    lat = np.linspace(np.pi / 2, -np.pi / 2, height)[:, None]     # +90 -> -90
    lon = np.linspace(-np.pi, np.pi, width)[None, :]

    # --- galactic plane, tilted so it does not line up with the disk -------
    # Steeply tilted, and with the galactic centre swung away from where the
    # camera points, so the band crosses frame as a diagonal streak instead
    # of filling it with haze.  The camera orbits during the shot, so this
    # also keeps the backdrop changing.
    tilt = np.deg2rad(58.0)
    node = np.deg2rad(-125.0)
    gal_lat = np.arcsin(
        np.clip(np.sin(lat) * np.cos(tilt)
                + np.cos(lat) * np.sin(tilt) * np.cos(lon - node), -1.0, 1.0))
    gal_lon = np.arctan2(
        np.cos(lat) * np.sin(lon - node),
        np.cos(lat) * np.cos(tilt) * np.cos(lon - node)
        - np.sin(lat) * np.sin(tilt))

    # Two-component band: a thin bright spine inside a broader halo, which is
    # roughly how the Milky Way reads by eye.
    band = (np.exp(-(gal_lat / 0.085) ** 2)
            + 0.28 * np.exp(-(gal_lat / 0.26) ** 2))
    # Brighten toward the galactic centre.
    band *= 0.26 + 0.74 * np.exp(-(gal_lon / 0.95) ** 2)

    # --- fractal structure -------------------------------------------------
    clouds = _fractal_field((height, width), beta=2.4, rng=rng)
    wisps = _fractal_field((height, width), beta=3.2, rng=rng)
    dust = _fractal_field((height, width), beta=2.1, rng=rng)

    # Mottle the band.  Clipped at zero so clouds can only subtract light,
    # never paint it outside the band.
    turbulence = np.clip(0.45 + 0.40 * clouds, 0.0, 1.6)

    # Dust lanes: only the densest ~16% of the dust field absorbs, giving
    # discrete dark lanes rather than an overall dimming.
    extinction = np.exp(-1.9 * np.clip(dust - 1.0, 0.0, None))

    glow = band * turbulence * extinction

    # High-latitude cirrus, restricted to the top few percent of the wisp
    # field so the rest of the sky stays genuinely black.
    cirrus = np.clip(wisps - 1.30, 0.0, None) * 0.05

    # --- colour ------------------------------------------------------------
    # Warm, dust-reddened core; cool blue-violet outskirts.
    warm = np.array([1.00, 0.72, 0.46])
    cool = np.array([0.34, 0.48, 1.00])
    mixv = np.clip(np.exp(-(gal_lat / 0.22) ** 2), 0.0, 1.0)[..., None]
    base = warm * mixv + cool * (1.0 - mixv)

    rgb = base * glow[..., None] * 0.80
    rgb += np.array([0.26, 0.34, 0.62]) * cirrus[..., None]
    rgb = np.clip(rgb, 0.0, None).astype(np.float32)

    img = bpy.data.images.new("BH_Nebula", width=width, height=height,
                              float_buffer=True)
    # Blender wants RGBA, bottom-up, flattened.
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = rgb[::-1]
    rgba[..., 3] = 1.0
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


# ---------------------------------------------------------------------------
# World shader
# ---------------------------------------------------------------------------

def _star_layer(nb: NodeBuilder, direction: Expr, scale: float, size: float,
                threshold: float, brightness: float) -> Expr:
    """
    One octave of stars.

    A Voronoi cell's F1 distance gives a point-like falloff around each seed,
    and the cell's random colour doubles as that star's magnitude and surface
    temperature -- so the field is deterministic, seamless in every direction,
    and infinitely sharp under lensing magnification.

    ``size`` is the stellar radius as a *fraction of a Voronoi cell*, which is
    the only scale-free way to express it: Blender reports F1 distance in the
    post-``Scale`` coordinate space, so a fixed distance threshold would make
    stars shrink to sub-pixel specks as soon as ``scale`` went up.
    """
    vor = nb.new("ShaderNodeTexVoronoi", direction.depth + 1)
    vor.voronoi_dimensions = "3D"
    vor.feature = "F1"
    nb.link(direction, vor.inputs["Vector"])
    vor.inputs["Scale"].default_value = scale
    vor.inputs["Randomness"].default_value = 1.0
    d = direction.depth + 1

    dist = Expr(nb, vor.outputs["Distance"], d)
    rand = Expr(nb, vor.outputs["Color"], d)
    rx, ry, rz = rand.xyz()

    # Point spread function: bright core, steep wings.
    core = (1.0 - (dist / size).min(1.0)) ** 3.0

    # Magnitude: a steep power on a uniform variate gives many faint stars and
    # a handful of bright ones, which is what the real luminosity function
    # looks like once you fold in distance.
    mag = ((rx - threshold).max(0.0) / max(1.0 - threshold, 1e-3)) ** 2.2

    # Colour temperature across roughly the M-to-B main sequence.
    bb = nb.new("ShaderNodeBlackbody", d + 1)
    nb.link(ry * 8600.0 + 3100.0, bb.inputs["Temperature"])
    tint = Expr(nb, bb.outputs["Color"], d + 1)

    return tint * (core * mag * brightness)


def build_world(cfg: Config) -> bpy.types.World:
    world = bpy.data.worlds.new("BH_Sky")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    nb = NodeBuilder(tree)

    tc = nb.new("ShaderNodeTexCoord", 0)
    direction = Expr(nb, tc.outputs["Generated"], 0)

    # --- stars -------------------------------------------------------------
    s = cfg.stars
    dens = s.star_density
    #                         scale          size   threshold  brightness
    stars = (_star_layer(nb, direction,  70.0 * dens, 0.17, 0.88, 2.2)
             + _star_layer(nb, direction, 170.0 * dens, 0.15, 0.90, 1.6)
             + _star_layer(nb, direction, 400.0 * dens, 0.13, 0.93, 0.8))
    stars = stars * s.star_brightness

    # --- nebula ------------------------------------------------------------
    env = nb.new("ShaderNodeTexEnvironment", 1)
    env.image = generate_nebula_image(cfg)
    nb.link(direction, env.inputs["Vector"])
    nebula = Expr(nb, env.outputs["Color"], 2) * s.nebula_strength

    bg = nb.new("ShaderNodeBackground", 30)
    nb.link(stars + nebula, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = s.sky_strength

    out = nb.new("ShaderNodeOutputWorld", 31)
    tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return world
