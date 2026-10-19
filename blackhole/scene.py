"""
Scene assembly and Cycles configuration (stages 1-5), plus the compositor.
"""

from __future__ import annotations

import bpy

from .config import Config
from . import camera as camera_mod
from . import disk as disk_mod
from . import lensing as lensing_mod
from . import starfield


# ---------------------------------------------------------------------------

def reset_scene() -> None:
    """Start from a genuinely empty file (the bpy module boots with a cube)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                       bpy.data.worlds, bpy.data.node_groups, bpy.data.actions):
        for item in list(collection):
            collection.remove(item)


def configure_cycles(cfg: Config) -> None:
    scene = bpy.context.scene
    r, lens = cfg.render, cfg.lensing

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.feature_set = "SUPPORTED"

    scene.render.resolution_x = r.resolution_x
    scene.render.resolution_y = r.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = r.fps
    scene.frame_start = 1
    scene.frame_end = cfg.frame_count
    scene.frame_step = 1

    scene.cycles.samples = r.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = r.adaptive_threshold
    scene.cycles.use_denoising = r.denoise
    if r.denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"

    # --- bounce budget -----------------------------------------------------
    # Every ray-portal crossing spends one *transparent* bounce, so this is
    # the dial that decides how far a photon may travel through the shell
    # stack -- and therefore how many times it can wind around the photon
    # sphere before Cycles gives up and returns black.
    scene.cycles.max_bounces = r.max_bounces
    scene.cycles.transparent_max_bounces = r.transparent_bounces
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0

    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 1.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.use_persistent_data = r.use_persistent_data

    scene.render.film_transparent = False
    scene.render.filter_size = 1.40

    # --- colour management -------------------------------------------------
    vs = scene.view_settings
    try:
        vs.view_transform = r.view_transform
    except TypeError:                                    # pragma: no cover
        vs.view_transform = "Filmic"
    try:
        vs.look = r.look
    except TypeError:                                    # pragma: no cover
        vs.look = "None"
    vs.exposure = r.exposure

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 15


def _vignette_image(cfg: Config) -> bpy.types.Image:
    """A smooth radial falloff matched to the render resolution."""
    import numpy as np

    w, h = cfg.render.resolution_x, cfg.render.resolution_y
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx + 0.5) / w * 2.0 - 1.0
    ny = ((yy + 0.5) / h * 2.0 - 1.0) * (h / w)
    radius = np.sqrt(nx * nx + ny * ny) / np.sqrt(1.0 + (h / w) ** 2)
    falloff = 1.0 - cfg.render.vignette * np.clip(radius, 0.0, 1.0) ** 2.6

    img = bpy.data.images.new("BH_Vignette", width=w, height=h,
                              float_buffer=True)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = falloff[::-1, :, None]
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def _set(node, name, value) -> bool:
    """
    Set a node parameter whether it lives on a socket or on the node.

    Blender keeps migrating compositor parameters from node properties onto
    input sockets (Glare, Blur and Lens Distortion all moved in 4.5).  Trying
    both keeps this file working across versions instead of pinning one.
    """
    sock = node.inputs.get(name)
    if sock is not None and hasattr(sock, "default_value"):
        try:
            sock.default_value = value
            return True
        except (TypeError, ValueError):
            pass
    attr = name.lower().replace(" ", "_")
    for candidate in (attr, f"use_{attr}"):
        if hasattr(node, candidate):
            try:
                setattr(node, candidate, value)
                return True
            except (TypeError, ValueError):
                pass
    return False


def build_compositor(cfg: Config) -> None:
    """Bloom, subtle chromatic aberration and a vignette."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    r = cfg.render

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (-600, 0)
    node = rl.outputs["Image"]
    x = -350

    # --- bloom -------------------------------------------------------------
    # The beamed inner disk is genuinely orders of magnitude brighter than the
    # sky, so veiling glare is lens physics here, not a cosmetic sparkle.
    glare = tree.nodes.new("CompositorNodeGlare")
    glare.location = (x, 120)
    try:
        glare.glare_type = "BLOOM"
    except TypeError:                                    # pragma: no cover
        glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    _set(glare, "Threshold", 1.0)
    _set(glare, "Strength", r.glare_strength)
    _set(glare, "Size", r.glare_size)
    _set(glare, "Smoothness", 0.4)
    tree.links.new(node, glare.inputs["Image"])
    node = glare.outputs["Image"]
    x += 220

    # --- a touch of lens dispersion ----------------------------------------
    disp = tree.nodes.new("CompositorNodeLensdist")
    disp.location = (x, 0)
    _set(disp, "Fit", True)
    _set(disp, "Distortion", 0.0)
    _set(disp, "Dispersion", 0.012)
    tree.links.new(node, disp.inputs["Image"])
    node = disp.outputs["Image"]
    x += 220

    # --- vignette ----------------------------------------------------------
    # Generated with NumPy rather than an EllipseMask + Blur chain: those
    # nodes keep changing API between versions and leave a visible hard edge
    # unless the blur radius is tuned to the resolution.  A float image is
    # exact, free at render time, and looks identical every run.
    if r.vignette > 0.0:
        img = tree.nodes.new("CompositorNodeImage")
        img.location = (x, -300)
        img.image = _vignette_image(cfg)

        mixn = tree.nodes.new("CompositorNodeMixRGB")
        mixn.location = (x + 260, 0)
        mixn.blend_type = "MULTIPLY"
        mixn.inputs["Fac"].default_value = 1.0
        tree.links.new(node, mixn.inputs[1])
        tree.links.new(img.outputs["Image"], mixn.inputs[2])
        node = mixn.outputs["Image"]
        x += 460

    comp = tree.nodes.new("CompositorNodeComposite")
    comp.location = (x, 0)
    tree.links.new(node, comp.inputs["Image"])


# ---------------------------------------------------------------------------

def build_scene(cfg: Config) -> dict:
    """
    Assemble the whole thing, in the documented stage order.

    Returns a dict of the created objects, handy for inspection and tests.
    """
    reset_scene()
    configure_cycles(cfg)

    scene = bpy.context.scene
    scene.world = starfield.build_world(cfg)                      # stage 4

    horizon = lensing_mod.build_event_horizon(cfg)                # stage 1
    cam, pivot = camera_mod.build_camera(cfg)                     # stage 1 + 5
    disk = disk_mod.build_accretion_disk(cfg)                     # stage 2
    shells, scheme = lensing_mod.build_lensing_shells(cfg)        # stage 3

    build_compositor(cfg)                                         # stage 6

    return dict(camera=cam, pivot=pivot, horizon=horizon,
                disk=disk, shells=shells, scheme=scheme)
