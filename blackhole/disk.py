"""
Stage 2 + 4 -- the accretion disk: thermodynamics, relativity, plasma.

The disk is built as a stack of thin annuli (an optically-thin emissive
medium).  Rays accumulate emission through every annulus they cross, and
because the lensing shells have already bent those rays, a single camera ray
often crosses the disk three or more times -- which is what produces the far
side of the disk arcing *over and under* the shadow, plus the thin higher
order images that make up the photon ring.  None of that is modelled by hand.

Physics applied per shading point:

  * **Novikov-Thorne temperature profile**
        T(r) ~ r^-3/4 [1 - sqrt(r_in/r)]^1/4
    with the no-torque inner boundary condition, so emission vanishes at the
    ISCO, peaks at r = (49/36) r_in, and falls off outward.

  * **Keplerian shear** -- the plasma pattern is sampled in a frame
    co-rotating at Omega(r) ~ r^-3/2, so turbulence winds up into spiral
    filaments exactly as differential rotation demands, and the pattern is
    additionally slowed by the gravitational time dilation factor
    sqrt(1 - 3M/r) so the innermost orbits visibly freeze.

  * **Relativistic Doppler + gravitational redshift**
        g = sqrt(1 - 1.5 r_s/r) / (1 - beta . n)
    The observed colour is a blackbody at ``g*T`` and the observed intensity
    is scaled by ``g^4``.  This is what makes the approaching side a searing
    blue-white and the receding side a dim red -- the single most recognisable
    signature of a real accretion disk, and the reason the EHT image of M87*
    is a crescent rather than a ring.
"""

from __future__ import annotations

import math

import bpy

from .config import Config
from .nodes import Expr, NodeBuilder
from .physics import kerr_isco_rs, novikov_thorne_temperature


# ---------------------------------------------------------------------------

def disk_inner_radius(cfg: Config) -> float:
    if cfg.physics.disk_inner is not None:
        return float(cfg.physics.disk_inner)
    return kerr_isco_rs(cfg.physics.spin)


def _fbm(nb: NodeBuilder, coord: Expr, scale: float, detail: float,
         roughness: float, distortion: float = 0.0) -> Expr:
    """A 4D noise lookup (the W input carries time for free animation)."""
    n = nb.new("ShaderNodeTexNoise", coord.depth + 1)
    n.noise_dimensions = "3D"
    nb.link(coord, n.inputs["Vector"])
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = roughness
    n.inputs["Distortion"].default_value = distortion
    return Expr(nb, n.outputs["Fac"], coord.depth + 1)


def build_disk_material(cfg: Config) -> bpy.types.Material:
    phys = cfg.physics
    r_in = disk_inner_radius(cfg)
    r_out = phys.disk_outer

    mat = bpy.data.materials.new("BH_AccretionDisk")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    nb = NodeBuilder(tree)

    # Animation clock.  A node can only be linked inside its own tree, so the
    # material owns its Value node and keyframes it directly; the result is
    # baked into the .blend and needs no drivers or script auto-execution.
    time_node = nb.new("ShaderNodeValue", 0)
    time_node.label = "Time (s)"
    last = cfg.frame_count
    for frame, value in ((1, 0.0), (last, cfg.render.duration_seconds)):
        time_node.outputs[0].default_value = value
        time_node.outputs[0].keyframe_insert("default_value", frame=frame)
    for fc in tree.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    time = Expr(nb, time_node.outputs[0], 0)

    pos = nb.geometry("Position")
    incoming = nb.geometry("Incoming")          # unit vector toward the camera
    px, py, pz = pos.xyz()

    # Cylindrical radius, and the planar unit vectors of the orbit.
    r_cyl = (px * px + py * py).sqrt().max(1e-4)
    r_hat_plane = nb.combine(px / r_cyl, py / r_cyl, 0.0)
    #  phi_hat = z_hat x r_hat  (prograde, co-rotating with the spin)
    phi_hat = nb.combine(-(py / r_cyl), px / r_cyl, 0.0)

    # =====================================================================
    # 1. Radial windows and vertical structure
    # =====================================================================
    # Soft inner/outer cutoffs so the disk does not end on a hard polygon.
    inner_edge = r_cyl.smoothstep(r_in, r_in * 1.22)
    outer_edge = 1.0 - r_cyl.smoothstep(r_out * 0.84, r_out)

    # Flared scale height H(r) and a Gaussian vertical density profile.
    height = (r_cyl / 10.0) ** phys.disk_flare * phys.disk_thickness
    z_norm = pz / height.max(1e-4)
    vertical = nb.math("EXPONENT", -(z_norm * z_norm))

    # =====================================================================
    # 2. Keplerian shear -> co-rotating sample frame
    # =====================================================================
    # Omega ~ r^-3/2, slowed by gravitational time dilation sqrt(1 - 3M/r)
    # so the inner disk appears to wind down rather than spin arbitrarily fast.
    dilation = (1.0 - 1.5 / r_cyl).max(0.0).sqrt()
    omega = dilation / (r_cyl ** 1.5) * 9.0
    angle = -(omega * time)

    rot = nb.new("ShaderNodeVectorRotate", angle.depth + 1)
    rot.rotation_type = "Z_AXIS"
    nb.link(pos, rot.inputs["Vector"])
    nb.link(angle, rot.inputs["Angle"])
    corot = Expr(nb, rot.outputs["Vector"], angle.depth + 1)

    # Domain warping: displacing the sample point by a low-frequency noise is
    # what turns bland cloud noise into something that reads as turbulent
    # magnetised plasma.
    warp = _fbm(nb, corot * 0.35, scale=1.0, detail=4.0, roughness=0.55)
    warped = corot + (warp - 0.5) * 2.4

    # Multi-scale density.  The azimuthal squash (x,y scaled differently from
    # z) stretches features into the orbital direction, giving filaments.
    squash = nb.combine(*[c for c in (warped.xyz())[:2]], (warped.xyz())[2] * 3.2)
    n1 = _fbm(nb, squash, scale=1.1, detail=8.0, roughness=0.62)
    n2 = _fbm(nb, squash * 3.1, scale=1.0, detail=6.0, roughness=0.55)
    n3 = _fbm(nb, squash * 8.7, scale=1.0, detail=4.0, roughness=0.50)
    density = n1 * 0.60 + n2 * 0.28 + n3 * 0.12

    # Contrast: push the field into bright filaments separated by voids.
    density = (density * 2.1 - 0.60).clamp01() ** 1.30

    # =====================================================================
    # 3. Novikov-Thorne temperature profile
    # =====================================================================
    x = (r_cyl / r_in).max(1.0)
    nt_shape = (x ** -0.75) * (1.0 - (1.0 / x).sqrt()).max(0.0) ** 0.25
    x_max = 49.0 / 36.0
    nt_norm = x_max ** -0.75 * (1.0 - math.sqrt(1.0 / x_max)) ** 0.25
    nt_shape = nt_shape / nt_norm                      # peaks at 1.0
    temperature = nt_shape * phys.peak_temperature

    # Radiated flux follows F ~ T^4, so this is the emissivity profile.
    emissivity = nt_shape ** 4

    # =====================================================================
    # 4. Relativistic Doppler beaming + gravitational redshift
    # =====================================================================
    # Local orbital speed of a circular Schwarzschild orbit: v = sqrt(M/(r-2M))
    beta = (0.5 / (r_cyl - 1.0).max(0.05)).sqrt().min(0.995)
    cos_angle = phi_hat.dot(incoming)                  # +1 = coming at us

    # g = sqrt(1 - 3M/r) / (1 - beta cos) -- exact for circular orbits, and it
    # correctly vanishes at the photon sphere where such orbits become null.
    g_grav = (1.0 - 1.5 / r_cyl).max(0.0).sqrt()
    g = (g_grav / (1.0 - beta * cos_angle).max(0.05)).min(8.0)

    # Observed colour: a blackbody at g*T.  Observed intensity: g^4 (bolometric
    # relativistic beaming).  Doing both is the physically consistent pair --
    # B_nu(T)/nu^3 is Lorentz invariant, so a blackbody stays a blackbody and
    # only its temperature and bolometric flux transform.
    bb = nb.new("ShaderNodeBlackbody", 30)
    nb.link((temperature * g).max(800.0).min(40000.0), bb.inputs["Temperature"])
    colour = Expr(nb, bb.outputs["Color"], 30)

    beamed = g ** phys.beaming_exponent

    # =====================================================================
    # 5. Assemble: emission + transparency = optically thin plasma
    # =====================================================================
    window = inner_edge * outer_edge * vertical
    alpha = (density * window * phys.disk_opacity).clamp01()

    strength = density * window * emissivity * beamed * phys.disk_brightness

    emit = nb.new("ShaderNodeEmission", 34)
    nb.link(colour, emit.inputs["Color"])
    nb.link(strength, emit.inputs["Strength"])

    transparent = nb.new("ShaderNodeBsdfTransparent", 34)
    mix = nb.new("ShaderNodeMixShader", 36)
    nb.link(alpha, mix.inputs["Fac"])
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(emit.outputs["Emission"], mix.inputs[2])

    out = nb.new("ShaderNodeOutputMaterial", 38)
    tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------

def build_accretion_disk(cfg: Config, layers: int = 9) -> bpy.types.Object:
    """
    Build the disk as ``layers`` stacked annuli spanning the scale height.

    Stacking flat annuli rather than using a true volume is a deliberate
    performance decision: Cycles volumes are enormously expensive when every
    camera ray already carries hundreds of transparent bounces from the
    lensing shells.  With the vertical Gaussian evaluated in the shader from
    the real z coordinate, a stack of annuli integrates to very nearly the
    same result at a small fraction of the cost.
    """
    import bmesh

    r_in = disk_inner_radius(cfg)
    r_out = cfg.physics.disk_outer
    h_ref = cfg.physics.disk_thickness

    bm = bmesh.new()
    seg, rings = 256, 96
    for i in range(layers):
        t = (i / (layers - 1)) * 2.0 - 1.0 if layers > 1 else 0.0
        scale_h = h_ref * (r_out / 10.0) ** cfg.physics.disk_flare
        z = t * 1.6 * scale_h
        verts = []
        for j in range(rings + 1):
            # Radial distribution biased inward, where the structure is.
            f = j / rings
            rad = r_in * (r_out / r_in) ** f
            row = []
            for k in range(seg):
                a = 2.0 * math.pi * k / seg
                row.append(bm.verts.new((rad * math.cos(a),
                                         rad * math.sin(a), z)))
            verts.append(row)
        for j in range(rings):
            for k in range(seg):
                k2 = (k + 1) % seg
                bm.faces.new((verts[j][k], verts[j][k2],
                              verts[j + 1][k2], verts[j + 1][k]))

    mesh = bpy.data.meshes.new("BH_AccretionDisk")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("BH_AccretionDisk", mesh)
    obj.data.materials.append(build_disk_material(cfg))
    bpy.context.collection.objects.link(obj)

    obj.visible_shadow = False
    obj.visible_diffuse = False
    obj.visible_glossy = False
    return obj
