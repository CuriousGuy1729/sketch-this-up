"""
Relativistic physics for the black-hole renderer.

Pure NumPy -- no Blender dependency -- so every formula here can be unit-tested
and validated independently of the renderer (see ``validate.py``).

=============================================================================
UNIT SYSTEM
=============================================================================
Geometric units with  G = c = 1  and  **1 Blender unit = 1 Schwarzschild radius
r_s = 2GM/c^2**.  Therefore, throughout this module:

    M       = 0.5          (black-hole mass)
    r_s     = 1.0          (event horizon of a non-rotating hole)
    r_ph    = 1.5          (photon sphere, a*=0)
    b_crit  = 3*sqrt(3)/2  = 2.598076...   (critical impact parameter / shadow)
    r_isco  = 3.0          (a*=0)

Radii named ``*_rs`` are in Schwarzschild radii; radii named ``*_m`` are in
gravitational radii r_g = GM/c^2 = r_s / 2 (the convention used by most of the
GR literature).  Conversion helpers are provided to keep the two straight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Fundamental constants of the unit system
# --------------------------------------------------------------------------
M_GEOM = 0.5                      # black-hole mass in units where r_s = 1
R_S = 1.0                         # Schwarzschild radius
R_PHOTON_SPHERE = 1.5             # photon sphere, Schwarzschild
B_CRIT = 1.5 * math.sqrt(3.0)     # 3*sqrt(3)/2 = 2.5980762...  (shadow radius)

# Physical constants (SI) -- only used for the "physical" temperature model
G_SI = 6.67430e-11
C_SI = 2.99792458e8
M_SUN = 1.98892e30
SIGMA_SB = 5.670374419e-8
STEFAN_EDD = 1.26e31              # Eddington luminosity per solar mass [W]


def rg_to_rs(r_m: float) -> float:
    """Convert gravitational radii (r_g = GM/c^2) to Schwarzschild radii."""
    return r_m * 0.5


def rs_to_rg(r_rs: float) -> float:
    """Convert Schwarzschild radii to gravitational radii."""
    return r_rs * 2.0


# =============================================================================
# 1.  KERR GEOMETRY -- characteristic radii
# =============================================================================

def kerr_horizon_rs(spin: float) -> float:
    """
    Outer event horizon of a Kerr black hole.

        r_+ = M + sqrt(M^2 - a^2)       [geometric units]

    With a = a* * M and M = 0.5 (so r_s = 1):

        r_+ / r_s = 0.5 * (1 + sqrt(1 - a*^2))

    a*=0 -> 1.0 (Schwarzschild);  a*=1 -> 0.5 (extremal).
    """
    spin = float(np.clip(spin, -0.9999, 0.9999))
    return 0.5 * (1.0 + math.sqrt(1.0 - spin * spin))


def kerr_isco_rs(spin: float) -> float:
    """
    Innermost Stable Circular Orbit (Bardeen, Press & Teukolsky 1972).

        Z1 = 1 + (1-a*^2)^(1/3) [ (1+a*)^(1/3) + (1-a*)^(1/3) ]
        Z2 = sqrt(3 a*^2 + Z1^2)
        r_isco/M = 3 + Z2 -/+ sqrt( (3-Z1)(3+Z1+2 Z2) )

    Minus sign = prograde (co-rotating) orbits, which is what an accretion
    disk aligned with the spin does.  Returned in Schwarzschild radii.

    Sanity values:  a*=0 -> 3.0 r_s (6 M);  a*=0.998 -> ~0.6 r_s (1.237 M).
    """
    a = float(np.clip(spin, -0.9999, 0.9999))
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    sign = -1.0 if a >= 0.0 else 1.0           # prograde for positive spin
    r_m = 3.0 + z2 + sign * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return rg_to_rs(r_m)


def kerr_photon_ring_rs(spin: float) -> float:
    """
    Radius of the prograde equatorial circular photon orbit:

        r_ph/M = 2 { 1 + cos[ (2/3) arccos(-|a*|) ] }     (prograde branch)

    a*=0 -> 3 M = 1.5 r_s;  a*=1 -> 1 M = 0.5 r_s.
    """
    a = float(np.clip(abs(spin), 0.0, 0.9999))
    r_m = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(-a)))
    return rg_to_rs(r_m)


def shadow_radius_rs(spin: float) -> float:
    """
    Apparent (lensed) shadow radius for an observer in the equatorial plane.

    For a*=0 this is exactly b_crit = 3*sqrt(3)/2 r_s.  A Kerr shadow is not a
    circle -- it is flattened on the prograde side -- so we return the mean
    areal radius, which stays remarkably close to the Schwarzschild value
    (it varies by only a few percent over the whole spin range).  Good enough
    for framing the camera and for validation tolerances.
    """
    a = float(np.clip(abs(spin), 0.0, 0.9999))
    # Empirical fit to the Kerr shadow mean radius (Johannsen & Psaltis 2010).
    return B_CRIT * (1.0 - 0.0045 * a ** 2 - 0.0225 * a ** 4)


# =============================================================================
# 2.  ORBITAL DYNAMICS OF THE DISK
# =============================================================================

def keplerian_omega(r_rs: np.ndarray | float, spin: float = 0.0) -> np.ndarray | float:
    """
    Coordinate angular velocity of a prograde circular equatorial orbit:

        Omega = 1 / ( r_M^{3/2} + a_M )      [geometric units, M=1]

    Reduces to the Newtonian/Schwarzschild sqrt(M/r^3) when a=0.  Returned in
    units of radians per unit coordinate time with r_s = 1 (so the orbital
    period at r is 2*pi/Omega).
    """
    r_m = rs_to_rg(np.asarray(r_rs, dtype=np.float64))
    a_m = float(spin)
    omega_m = 1.0 / (np.power(np.maximum(r_m, 1e-6), 1.5) + a_m)
    # Omega has dimensions 1/length -> convert from 1/r_g to 1/r_s
    return omega_m * 2.0


def orbital_beta(r_rs: np.ndarray | float) -> np.ndarray | float:
    """
    Orbital speed of a circular Schwarzschild orbit as measured by a *static
    local observer*, in units of c:

        v = sqrt( M / (r - 2M) )    ->    sqrt( 0.5 / (r_rs - 1) )

    Checks:  r = 1.5 r_s (photon sphere) -> v = 1 (light speed, as required)
             r = 3   r_s (ISCO, a*=0)    -> v = 0.5 c   (textbook value)
    """
    r = np.asarray(r_rs, dtype=np.float64)
    return np.sqrt(0.5 / np.maximum(r - 1.0, 1e-6))


def redshift_factor(r_rs, cos_angle, clamp=True):
    """
    Combined relativistic redshift / Doppler factor ``g = nu_obs / nu_emit``
    for an emitter on a prograde circular orbit, seen by a distant observer:

        g = sqrt(1 - 3M/r) / (1 - beta . n_hat)
          = sqrt(1 - 1.5/r_rs) / (1 - beta * cos_angle)

    The numerator bundles gravitational redshift *and* orbital time dilation
    (the standard exact Schwarzschild result -- it correctly vanishes at the
    photon sphere, where circular orbits become null).  The denominator is the
    special-relativistic Doppler shift; ``cos_angle`` is the cosine of the
    angle between the orbital velocity and the direction to the observer.

    Observed brightness then follows the relativistic beaming law

        I_obs = g^4 * I_emit                 (bolometric)

    and the observed colour is a blackbody at temperature  g * T_emit.
    """
    r = np.asarray(r_rs, dtype=np.float64)
    beta = orbital_beta(r)
    grav = np.sqrt(np.maximum(1.0 - 1.5 / r, 0.0))
    denom = 1.0 - beta * np.asarray(cos_angle, dtype=np.float64)
    g = grav / np.maximum(denom, 1e-4)
    return np.clip(g, 0.0, 12.0) if clamp else g


# =============================================================================
# 3.  THERMAL STRUCTURE -- Shakura-Sunyaev / Novikov-Thorne thin disk
# =============================================================================

def novikov_thorne_temperature(r_rs, r_in_rs, t_peak_k):
    """
    Effective temperature profile of a radiatively efficient thin disk:

        T(r) proportional to  r^{-3/4} * [ 1 - sqrt(r_in / r) ]^{1/4}

    The bracket is the no-torque inner boundary condition: emission goes to
    zero at the inner edge, rises to a maximum at r = (49/36) r_in, then falls
    off as r^{-3/4}.  The profile is renormalised so that its peak equals
    ``t_peak_k`` kelvin, which is the knob the artist actually wants.
    """
    r = np.asarray(r_rs, dtype=np.float64)
    x = np.maximum(r / r_in_rs, 1.0)
    shape = np.power(x, -0.75) * np.power(np.maximum(1.0 - np.sqrt(1.0 / x), 0.0), 0.25)
    # Analytic maximum of the shape function occurs at x = 49/36.
    x_max = 49.0 / 36.0
    shape_max = x_max ** -0.75 * (1.0 - math.sqrt(1.0 / x_max)) ** 0.25
    return t_peak_k * shape / shape_max


def physical_peak_temperature(mass_solar: float, eddington_ratio: float, spin: float) -> float:
    """
    Peak effective temperature of a Novikov-Thorne disk derived from first
    principles, for the ``--physical-temperature`` mode.

        L     = eddington_ratio * 1.26e31 * (M/Msun)   [W]
        A     ~ 4*pi*r_in^2                            [emitting area scale]
        T_eff = (L / (A * sigma))^{1/4}

    This is a scaling estimate, not a full NT integration.  A 10^7 Msun hole at
    10% Eddington lands around 10^5 K (peaking in the far UV): physically
    correct, but it renders as a featureless blue-white -- which is exactly why
    the cinematic default remaps the colour temperature instead.
    """
    r_in_rs = kerr_isco_rs(spin)
    r_s_m = 2.0 * G_SI * mass_solar * M_SUN / (C_SI ** 2)
    r_in_m = r_in_rs * r_s_m
    lum = eddington_ratio * STEFAN_EDD * mass_solar
    area = 4.0 * math.pi * r_in_m ** 2
    return float((lum / (area * SIGMA_SB)) ** 0.25)


# =============================================================================
# 4.  NULL GEODESICS -- the lensing engine
# =============================================================================
#
# Photon motion in Schwarzschild spacetime, written in Cartesian form, obeys
#
#       d^2 x/dL^2 = -(3/2) * r_s * h^2 * x / r^5 ,      h = |x cross v|
#
# (the Cartesian equivalent of the Binet equation d^2u/dphi^2 + u = 3 M u^2).
# With r_s = 1 the coefficient is exactly 1.5.  ``h`` is the conserved specific
# angular momentum, numerically equal to the impact parameter of the current
# straight-line segment when |v| = 1.
#
# Kerr frame dragging is added as a Lense-Thirring rotation of the ray
# direction at angular rate
#
#       omega_LT = 2 J / r^3 = 0.5 * a* / r^3           (spin along +z)
#
# which integrates to a total azimuthal deflection of a*/b^2 -- exactly the
# known gravitomagnetic light deflection 4GJ/(c^3 b^2).  (See ``validate.py``.)
# =============================================================================


def geodesic_acceleration(pos: np.ndarray, vel: np.ndarray, spin: float = 0.0) -> np.ndarray:
    """
    Photon 'acceleration' d^2x/dL^2 at position ``pos`` with direction ``vel``.

    ``pos`` and ``vel`` are (..., 3) arrays.  Returns the same shape.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    r = np.linalg.norm(pos, axis=-1, keepdims=True)
    r = np.maximum(r, 1e-9)
    h_vec = np.cross(pos, vel)
    h2 = np.sum(h_vec * h_vec, axis=-1, keepdims=True)
    acc = -1.5 * h2 * pos / np.power(r, 5)
    if spin:
        # Lense-Thirring: omega x v, with omega = 0.5*a*/r^3 along +z.
        omega = np.zeros_like(pos)
        omega[..., 2] = (0.5 * spin) / np.power(r[..., 0], 3)
        acc = acc + np.cross(omega, vel)
    return acc


def trace_geodesic_rk4(b: float, spin: float = 0.0, r_start: float = 1000.0,
                       steps: int = 200000, d_lambda: float = 0.01,
                       r_capture: float = 1.0, r_escape: float | None = None):
    """
    Reference integrator: adaptive-ish RK4 of the exact null geodesic.

    A photon starts at x = (-r_start, b, 0) travelling along +x, i.e. with
    impact parameter ``b``.  Integration stops when the photon is captured
    (r < r_capture) or escapes back beyond ``r_escape``.

    Returns ``(captured, deflection_angle, r_min)`` where ``deflection_angle``
    is the total angle between the incoming and outgoing direction in radians
    (0 = undeflected, pi = sent straight back).

    This is the ground truth the fast shell integrator is validated against.
    """
    if r_escape is None:
        r_escape = r_start * 1.05

    pos = np.array([-r_start, b, 0.0], dtype=np.float64)
    vel = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_in = vel.copy()
    r_min = r_start

    for _ in range(steps):
        r = float(np.linalg.norm(pos))
        r_min = min(r_min, r)
        if r < r_capture:
            return True, float("nan"), r_min
        if r > r_escape and float(np.dot(pos, vel)) > 0.0:
            break

        # Step size shrinks near the hole where curvature is strong.
        h = d_lambda * max(min(r, 50.0) ** 1.5 * 0.05, 0.002)

        k1v = geodesic_acceleration(pos, vel, spin)
        k1x = vel
        k2v = geodesic_acceleration(pos + 0.5 * h * k1x, vel + 0.5 * h * k1v, spin)
        k2x = vel + 0.5 * h * k1v
        k3v = geodesic_acceleration(pos + 0.5 * h * k2x, vel + 0.5 * h * k2v, spin)
        k3x = vel + 0.5 * h * k2v
        k4v = geodesic_acceleration(pos + h * k3x, vel + h * k3v, spin)
        k4x = vel + h * k3v

        pos = pos + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        vel = vel + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        vel /= np.linalg.norm(vel)

    cos_d = float(np.clip(np.dot(v_in, vel), -1.0, 1.0))
    return False, math.acos(cos_d), r_min


# ---------------------------------------------------------------------------
# The discrete "nested shell" integrator -- a faithful CPU replica of what the
# Ray Portal shader does inside Cycles.  Keeping the two in lockstep is what
# lets us certify the renderer's lensing numerically.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShellScheme:
    """
    Radii + step algebra for the nested ray-portal shells.

    Shells are placed uniformly in the variable ``s = r**exponent`` (with
    ``exponent -> 0`` meaning ``s = ln r``).  The exponent controls where
    resolution is spent:

        exponent =  0.0  -> logarithmic:  dr/r constant
        exponent = -0.5  -> dr ~ r^1.5 :  denser near the hole
        exponent = -1.0  -> dr ~ r^2   :  densest near the hole, coarse outside

    Negative exponents concentrate shells around the photon sphere (r = 1.5),
    which is where trajectories are chaotically sensitive and where the shadow
    radius is decided.  ``-0.5`` is the calibrated optimum -- see validate.py.

    The crucial property is that the neighbours of a shell have a *closed
    form*, so the shader can work out its own step length from ``r`` alone
    without any per-shell data:

        r_inner = (r^p + ds)^(1/p),     r_outer = (r^p - ds)^(1/p)
    """
    r_min: float
    r_max: float
    count: int
    exponent: float = -0.5

    @property
    def is_log(self) -> bool:
        return abs(self.exponent) < 1e-9

    @property
    def delta_s(self) -> float:
        """Uniform increment in ``s`` between adjacent shells."""
        if self.is_log:
            return math.log(self.r_max / self.r_min) / (self.count - 1)
        p = self.exponent
        return abs(self.r_min ** p - self.r_max ** p) / (self.count - 1)

    def radii(self) -> np.ndarray:
        """Shell radii, outermost first."""
        i = np.arange(self.count)
        if self.is_log:
            r = self.r_min * np.exp(self.delta_s * i)
        else:
            p = self.exponent
            s = self.r_min ** p - self.delta_s * i      # p<0 -> s decreases outward
            r = np.power(np.maximum(s, 1e-12), 1.0 / p)
        return np.sort(r)[::-1].copy()

    def neighbours(self, r: float) -> tuple[float, float]:
        """Radii of the shells immediately inside and outside ``r``."""
        ds = self.delta_s
        if self.is_log:
            return r * math.exp(-ds), r * math.exp(ds)
        p = self.exponent
        rp = r ** p
        r_in = max(rp + ds, 1e-12) ** (1.0 / p)
        r_out = max(rp - ds, 1e-12) ** (1.0 / p)
        return r_in, r_out


def shell_step_length(r: float, mu_signed: float, h2: float,
                      r_in: float, r_out: float) -> float:
    """
    Exact straight-line distance from a point on shell ``r`` to the next shell.

    This is the heart of the integrator's accuracy.  A naive estimate uses the
    radial gap, ``ds = r*coeff/|mu|``, but that diverges at a turning point and
    badly misweights the pericenter -- precisely where the deflection is
    generated.  Solving the ray/sphere intersection in closed form instead:

    For a ray  P + t*v  with  r = |P|,  mu = v.r_hat,  h^2 = |P x v|^2,
    the distance to a sphere of radius R is  t = -r*mu + sqrt(R^2 - h^2).

      * Travelling inward (mu < 0) toward ``r_in``:
            - if h < r_in the ray reaches it:  ds = r|mu| - sqrt(r_in^2 - h^2)
            - otherwise the ray turns around inside this gap and re-crosses the
              *current* shell after a chord of  ds = 2 r |mu|
      * Travelling outward (mu > 0) toward ``r_out``:
            ds = sqrt(r_out^2 - h^2) - r*mu

    Every branch is finite, exact, and expressible in shader nodes.
    """
    r_mu = r * mu_signed
    if mu_signed < 0.0:
        disc = r_in * r_in - h2
        if disc > 0.0:
            return -r_mu - math.sqrt(disc)          # reaches the next shell in
        return -2.0 * r_mu                          # turns around: chord back
    return math.sqrt(max(r_out * r_out - h2, 0.0)) - r_mu


def is_doomed(pos: np.ndarray, vel: np.ndarray) -> bool:
    """
    Exact capture test -- is this photon already guaranteed to fall in?

    Schwarzschild null geodesics have a turning point if and only if their
    impact parameter exceeds b_crit = 3*sqrt(3)/2 r_s.  So an **inward-moving**
    photon with b < b_crit can never turn around: it is captured, full stop.
    No integration required.

    The subtlety is that ``h = |P x v|`` is *not* b.  With v a unit 3-vector in
    these coordinates, h is the local perpendicular distance, related to the
    conserved impact parameter by

        b = h / sqrt(1 - r_s/r)

    (h therefore shrinks as the photon falls in, exactly as observed).

    Applying this at every shell crossing does two things: it makes the shadow
    edge land at precisely 2.598 r_s instead of wherever the discrete scheme's
    chaotic separatrix happens to fall, and it kills the doomed rays
    immediately rather than letting them spiral for a hundred portal bounces --
    which is also where most of the render time was going.
    """
    r = max(float(np.linalg.norm(pos)), 1e-9)
    if float(np.dot(vel, pos)) >= 0.0:          # moving outward: leave it be
        return False
    h = float(np.linalg.norm(np.cross(pos, vel)))
    lapse = math.sqrt(max(1.0 - 1.0 / r, 1e-12))
    return (h / lapse) < B_CRIT


def portal_kick(pos: np.ndarray, vel: np.ndarray, scheme: 'ShellScheme',
                spin: float = 0.0, midpoint: bool = True) -> np.ndarray:
    """
    The complete per-shell operation, mirrored 1:1 by the Ray Portal shader.

    This is a **kick-drift leapfrog**.  The subtlety that makes or breaks the
    scheme: the kick applied at a shell must be weighted by the length of the
    segment that *follows* it, but the ray travels that segment along the
    post-kick direction, which is not known until the kick is computed.

    A single predictor-corrector pass resolves the circularity:

        1. predict  ds  from the incoming direction, get a trial kick
        2. recompute ds from the trial direction and apply the final kick

    After this, ``ds`` matches the distance Cycles actually traces to the next
    shell to ~1e-15, so each kick carries exactly the right path weight.
    Without it, the scheme over-deflects catastrophically in the strong field
    (rays at b = 4 r_s get spuriously captured).

    Returns the new unit direction.
    """
    pos = np.asarray(pos, dtype=np.float64)
    v_in = np.asarray(vel, dtype=np.float64)
    r = max(float(np.linalg.norm(pos)), 1e-9)
    r_hat = pos / r
    r_in, r_out = scheme.neighbours(r)

    def _impulse(direction: np.ndarray) -> np.ndarray:
        """Midpoint-quadrature impulse of the segment leaving ``pos`` along
        ``direction`` (used for both the forward and the backward segment)."""
        h2 = float(np.dot(np.cross(pos, direction), np.cross(pos, direction)))
        mu_dir = float(np.dot(direction, r_hat))
        ds = shell_step_length(r, mu_dir, h2, r_in, r_out)

        def _accel_at(point: np.ndarray) -> np.ndarray:
            rr = max(float(np.linalg.norm(point)), 1e-9)
            a = -1.5 * h2 * (point / rr) / (rr ** 4)
            if spin:
                # Lense-Thirring frame dragging: omega x v,
                # omega = 0.5 a* / r^3 along the spin axis (+z).
                omega = np.array([0.0, 0.0, (0.5 * spin) / rr ** 3])
                a = a + np.cross(omega, v_in)
            return a

        if not midpoint:                       # 1st-order reference scheme
            return _accel_at(pos) * ds
        # Simpson's rule along the segment.  A single midpoint sample is not
        # enough at the pericenter: there the ray turns around inside one gap,
        # so the chord reaches ~2r*sqrt(2 dr/r) and |a| ~ r^-4 varies by tens
        # of percent across it.  Three samples make the quadrature 4th order
        # and cost two extra evaluations of a very cheap expression.
        a0 = _accel_at(pos)
        a1 = _accel_at(pos + direction * (0.5 * ds))
        a2 = _accel_at(pos + direction * ds)
        return (a0 + 4.0 * a1 + a2) * (ds / 6.0)

    # Symmetric (leapfrog / Stoermer-Verlet) impulse splitting.  Each segment's
    # total kick is shared equally between the crossings at its two ends, so
    # this crossing carries half of the segment behind it and half of the
    # segment ahead.  Summing over crossings telescopes to the full path, and
    # the scheme is time-reversible -- essential at the pericenter, where the
    # ray turns around inside a single gap and the chord can reach ~45% of r.
    # Kicking only at the start of that chord (the naive scheme) rotates the
    # ray too early and makes it dive far too deep.
    #
    # The backward half is fixed by the direction we arrived along; the forward
    # half must be measured along the direction we will *leave* along, which is
    # not known until the kick is applied.  One corrector pass closes that loop
    # (without it the strong field over-deflects and swallows rays out to
    # b = 4 r_s).
    dv_back = _impulse(-v_in)
    v_out = v_in
    for _ in range(2):
        dv = 0.5 * (_impulse(v_out) + dv_back)
        # A photon always travels at c, so the impulse may only *rotate* v.
        # Projecting out the component along v keeps |v| = 1 to second order
        # and stops the scheme bleeding the kick into the radial direction.
        dv = dv - np.dot(dv, v_in) * v_in
        v_out = v_in + dv
        v_out = v_out / np.linalg.norm(v_out)
    return v_out


def trace_geodesic_shells(b: float, scheme: ShellScheme, spin: float = 0.0,
                          midpoint: bool = True, max_crossings: int = 4096,
                          r_capture: float = 1.0):
    """
    Integrate a photon using *exactly* the algorithm the shader implements.

    At every shell crossing:

        ds   = shell_step_length(r, mu, h^2, r_in, r_out)   (exact)
        P_m  = P + v*ds/2        -- midpoint sample, makes the scheme 2nd order
        dv   = -1.5 * h^2 * r_hat(P_m) / |P_m|^4 * ds    (+ Lense-Thirring)
        v   <- normalize(v + dv)

    and the ray then travels in a straight line to the next shell.  This
    function exists so ``validate.py`` can certify the renderer's lensing
    against the RK4 ground truth without ever launching Blender.

    Returns ``(captured, deflection_angle, r_min, crossings)``.
    """
    shell_radii = np.sort(scheme.radii())    # ascending
    r_outer = float(shell_radii[-1])
    # The Blender scene contains a solid absorbing sphere at the horizon in
    # addition to the portal shells; include it here so capture is detected by
    # real intersection rather than by sampling r only at shell crossings.
    radii = np.sort(np.append(shell_radii, r_capture))

    # Start just outside the shell stack, aimed with impact parameter b.
    x0 = math.sqrt(max(r_outer ** 2 - b ** 2, 1e-6))
    pos = np.array([-x0, b, 0.0], dtype=np.float64)
    vel = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_in = vel.copy()
    r_min_seen = r_outer

    for crossing in range(max_crossings):
        r = float(np.linalg.norm(pos))
        r_min_seen = min(r_min_seen, r)
        if r < r_capture or is_doomed(pos, vel):
            return True, float("nan"), r_min_seen, crossing
        if r > r_outer * 1.0001 and float(np.dot(pos, vel)) > 0.0:
            cos_d = float(np.clip(np.dot(v_in, vel), -1.0, 1.0))
            return False, math.acos(cos_d), r_min_seen, crossing

        # --- the portal kick (identical to the shader) ---------------------
        vel = portal_kick(pos, vel, scheme, spin, midpoint)

        # --- travel to the next shell intersection -------------------------
        t = _next_shell_hit(pos, vel, radii)
        if t is None:
            cos_d = float(np.clip(np.dot(v_in, vel), -1.0, 1.0))
            return False, math.acos(cos_d), r_min_seen, crossing
        pos = pos + vel * t
        if float(np.linalg.norm(pos)) <= r_capture * 1.0001:
            return True, float("nan"), r_capture, crossing

    return True, float("nan"), r_min_seen, max_crossings


def _next_shell_hit(pos: np.ndarray, vel: np.ndarray, radii: np.ndarray,
                    eps: float = 1e-6) -> float | None:
    """Smallest positive ray/sphere intersection distance over all shells."""
    b_dot = float(np.dot(pos, vel))
    p2 = float(np.dot(pos, pos))
    disc = b_dot * b_dot - (p2 - radii * radii)
    best = None
    for rad_i, d_i in zip(radii, disc):
        if d_i <= 0.0:
            continue
        sq = math.sqrt(d_i)
        for t in (-b_dot - sq, -b_dot + sq):
            if t > eps and (best is None or t < best):
                best = t
    return best


def measure_critical_impact_parameter(trace_fn, lo: float = 1.0, hi: float = 6.0,
                                      iterations: int = 48) -> float:
    """
    Bisect for the capture threshold b_crit of any tracer.

    ``trace_fn(b)`` must return a tuple whose first element is ``captured``.
    The exact Schwarzschild answer is 3*sqrt(3)/2 = 2.5980762...
    """
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if trace_fn(mid)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def weak_field_deflection(b: float, order: int = 2) -> float:
    """
    Post-Newtonian light deflection for a Schwarzschild lens.

        alpha = 2 r_s / b  +  (15 pi / 16) (r_s / b)^2  +  O((r_s/b)^3)

    The leading term is Einstein's 4GM/bc^2.  The second-order term matters
    more than one might expect: at b = 12 r_s it is already 12% of the total,
    so comparing a numerical integrator against the first-order formula alone
    would wrongly flag a correct integrator as broken.
    """
    x = R_S / b
    alpha = 2.0 * x
    if order >= 2:
        alpha += (15.0 * math.pi / 16.0) * x * x
    return alpha
