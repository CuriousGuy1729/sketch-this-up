"""
Stage 3 -- gravitational lensing via nested Ray Portal shells.

Cycles traces straight lines.  To bend light we exploit the **Ray Portal BSDF**
(Blender 4.2+), which lets a shader rewrite a ray's position and direction and
continue tracing.  Wrapping the black hole in a stack of concentric spheres,
each carrying a portal that applies one integration step, turns Cycles into a
geodesic ray-tracer: the geometry supplies the step boundaries, the shader
supplies the physics.

Crucially this means *everything* is lensed for free and consistently -- the
accretion disk, the star field, and any object you add -- because they are
simply intersected by rays that are already following curved paths.  Multiple
disk crossings per ray produce the photon ring and the "over and under"
Interstellar silhouette without any of it being faked.

The algorithm implemented below is identical, line for line, to
``physics.portal_kick``; ``validate.py`` certifies the NumPy twin against an
RK4 integration of the exact null geodesic.  See that module for the
derivations and the accuracy report.

Transparent bounces are the budget here: each shell crossing consumes one, so
``render.transparent_bounces`` must comfortably exceed 2 x shell count.
"""

from __future__ import annotations

import math

import bpy

from .config import Config
from .nodes import Expr, NodeBuilder
from .physics import B_CRIT, ShellScheme


# ---------------------------------------------------------------------------

def build_shell_scheme(cfg: Config) -> ShellScheme:
    lens = cfg.lensing
    return ShellScheme(r_min=lens.r_min, r_max=lens.r_max,
                       count=lens.shells, exponent=lens.spacing_exponent)


# ---------------------------------------------------------------------------

def _simpson_impulse(nb: NodeBuilder, pos: Expr, direction: Expr,
                     r: Expr, r_hat: Expr, r_in: Expr, r_out: Expr,
                     spin: float, gain: float) -> Expr:
    """
    Impulse delivered by the segment that leaves ``pos`` along ``direction``.

    Mirrors ``physics.portal_kick._impulse``:

      * exact ray/sphere step length to the adjacent shell (three branches),
      * Simpson quadrature of the acceleration along that segment,
      * optional Lense-Thirring frame dragging.
    """
    # --- conserved quantities of this straight segment ---------------------
    h_vec = pos.cross(direction)
    h2 = h_vec.dot(h_vec)
    mu = direction.dot(r_hat)                       # signed radial cosine
    r_mu = r * mu

    # --- exact distance to the next shell ----------------------------------
    # inward branch, reaching the next shell in:   -r*mu - sqrt(r_in^2 - h^2)
    disc_in = r_in * r_in - h2
    ds_reach = -r_mu - disc_in.max(0.0).sqrt()
    # inward branch, turning around inside the gap: chord back to this shell
    ds_turn = r_mu * -2.0
    ds_inward = nb.select(disc_in.gt(0.0), ds_reach, ds_turn)
    # outward branch
    ds_outward = (r_out * r_out - h2).max(0.0).sqrt() - r_mu
    ds = nb.select(mu.lt(0.0), ds_inward, ds_outward)
    ds = ds.max(0.0)

    # --- Simpson quadrature of a(s) over [0, ds] ---------------------------
    def accel_at(point: Expr) -> Expr:
        rr = point.length().max(1e-6)
        a = point * (h2 * (-1.5 * gain) / (rr ** 4 * rr))   # = -1.5 h^2 P/r^5
        if spin:
            # omega x v, with omega = 0.5 a* / r^3 about the spin axis (+Z).
            omega = nb.combine(0.0, 0.0, (0.5 * spin) / (rr ** 3))
            a = a + omega.cross(direction)
        return a

    a0 = accel_at(pos)
    a1 = accel_at(pos + direction * (ds * 0.5))
    a2 = accel_at(pos + direction * ds)
    return (a0 + a1 * 4.0 + a2) * (ds * (1.0 / 6.0))


def build_lensing_material(cfg: Config, scheme: ShellScheme) -> bpy.types.Material:
    """Create the Ray Portal material shared by every shell."""
    lens = cfg.lensing
    spin = cfg.physics.spin if lens.frame_dragging else 0.0
    gain = lens.deflection_gain
    ds_log = scheme.delta_s

    mat = bpy.data.materials.new("BH_GeodesicPortal")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    nb = NodeBuilder(tree)

    # --- ray state ---------------------------------------------------------
    # The shells are centred on the world origin, so world space *is*
    # black-hole-centric space and Position needs no transform.
    pos = nb.geometry("Position")
    v_in = (-nb.geometry("Incoming")).normalize()

    r = pos.length().max(1e-6)
    r_hat = pos / r

    # Neighbouring shell radii, closed form (see ShellScheme.neighbours).
    if scheme.is_log:
        r_in = r * math.exp(-ds_log)
        r_out = r * math.exp(ds_log)
    else:
        p = scheme.exponent
        rp = r ** p
        r_in = (rp + ds_log).max(1e-9) ** (1.0 / p)
        r_out = (rp - ds_log).max(1e-9) ** (1.0 / p)

    def impulse(direction: Expr) -> Expr:
        return _simpson_impulse(nb, pos, direction, r, r_hat,
                                r_in, r_out, spin, gain)

    # --- symmetric leapfrog with one corrector pass ------------------------
    # Half of the segment behind us, half of the segment ahead.  The forward
    # half must be measured along the direction we leave with, so we iterate
    # once.  (Both refinements are essential: see physics.portal_kick.)
    dv_back = impulse(-v_in)
    v_out = v_in
    for _ in range(2):
        dv = (impulse(v_out) + dv_back) * 0.5
        dv = dv - v_in * dv.dot(v_in)        # photons only turn, never speed up
        v_out = (v_in + dv).normalize()

    # --- exact capture test ------------------------------------------------
    # A Schwarzschild photon has a turning point iff its impact parameter
    # exceeds b_crit = 3*sqrt(3)/2.  An inward-moving ray below that is
    # already doomed, so we can absorb it here analytically instead of
    # integrating it down a spiral.  This pins the shadow edge to exactly
    # 2.598 r_s -- otherwise its radius is decided by wherever the discrete
    # scheme's chaotic separatrix happens to land -- and it reclaims the
    # hundreds of transparent bounces those rays used to burn.
    h_in = pos.cross(v_in).length()
    lapse = (1.0 - 1.0 / r).max(1e-6).sqrt()
    b_true = h_in / lapse
    inbound = v_in.dot(pos).lt(0.0)
    doomed = inbound * b_true.lt(B_CRIT)

    # --- hand the bent ray back to Cycles ----------------------------------
    portal = nb.new("ShaderNodeBsdfRayPortal", 40)
    portal.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    # Nudge off the surface along the new direction so the ray cannot
    # immediately re-intersect the shell it is leaving.
    nb.link(pos + v_out * 1e-4, portal.inputs["Position"])
    nb.link(v_out, portal.inputs["Direction"])

    # Absorbing branch: zero-strength emission terminates the path with no
    # radiance, which *is* the shadow.
    absorb = nb.new("ShaderNodeEmission", 40)
    absorb.inputs["Strength"].default_value = 0.0
    absorb.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

    swap = nb.new("ShaderNodeMixShader", 41)
    nb.link(doomed, swap.inputs["Fac"])
    tree.links.new(portal.outputs["BSDF"], swap.inputs[1])
    tree.links.new(absorb.outputs["Emission"], swap.inputs[2])

    out = nb.new("ShaderNodeOutputMaterial", 42)
    tree.links.new(swap.outputs["Shader"], out.inputs["Surface"])
    return mat


# ---------------------------------------------------------------------------

def build_lensing_shells(cfg: Config) -> tuple[bpy.types.Object, ShellScheme]:
    """
    Create the shell stack as a single mesh object.

    All shells live in one object (and share one material) so Cycles sees one
    BVH and one shader -- far cheaper than N objects, and the shader derives
    everything it needs from |P| alone, so the shells need no per-shell data.
    """
    import bmesh

    scheme = build_shell_scheme(cfg)
    radii = scheme.radii()

    bm = bmesh.new()
    for radius in radii:
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=cfg.lensing.sphere_subdivisions,
            radius=float(radius),
        )

    mesh = bpy.data.meshes.new("BH_LensingShells")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("BH_LensingShells", mesh)
    obj.data.materials.append(build_lensing_material(cfg, scheme))
    bpy.context.collection.objects.link(obj)

    # The shells exist only to bend camera-side rays.  Hiding them from every
    # other ray type keeps the BVH queries down and stops them interfering
    # with light sampling.
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False
    obj.visible_shadow = False
    obj.cycles.use_adaptive_subdivision = False
    return obj, scheme


def build_event_horizon(cfg: Config) -> bpy.types.Object:
    """
    The absorbing surface at the Kerr horizon.

    Rays that reach it are terminated with zero radiance -- that *is* the
    shadow.  An Emission shader of strength 0 is the cleanest way to say
    "this path contributes nothing and does not continue".

    Note the shadow you see in the render is ~2.6 r_s across, not the 1.0 r_s
    of this sphere: the apparent size is set by the critical impact parameter
    b_crit = 3*sqrt(3)/2, and it emerges from the lensing rather than being
    modelled.
    """
    from .physics import kerr_horizon_rs

    radius = kerr_horizon_rs(cfg.physics.spin)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48,
                                         radius=radius, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "BH_EventHorizon"
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new("BH_Horizon")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 0.0
    emit.inputs["Color"].default_value = (0, 0, 0, 1)
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    out.location = (220, 0)
    tree.links.new(emit.outputs[0], out.inputs["Surface"])

    obj.data.materials.append(mat)
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    obj.visible_volume_scatter = False
    return obj
