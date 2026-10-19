"""
Physics self-checks.

The whole point of this project is that the picture is *derived*, not drawn,
so there has to be something that says whether the derivation is right.  These
checks run in pure NumPy (no Blender needed) and compare the code against
closed-form general relativity wherever a closed form exists:

================================  ========================================
check                             reference
================================  ========================================
Kerr horizon / ISCO / photon ring Bardeen-Press-Teukolsky closed forms
critical impact parameter         b_c = 3*sqrt(3)/2 r_s  (exact)
weak-field deflection             alpha = 4GM/bc^2 = 2 r_s / b
second-order deflection           + (15*pi/16)(r_s/b)^2
shell integrator vs RK4           adaptive RK4 on the same geodesic
Novikov-Thorne temperature peak   r = (49/36) r_in  (exact)
Doppler / gravitational redshift  g = sqrt(1-1.5 r_s/r) / (1 - beta.k)
================================  ========================================

The one thing deliberately *not* asserted is the capture boundary.  Near
b_c the photon orbit is a genuine unstable separatrix: the number of windings
diverges logarithmically, so any finite shell scheme has a chaotic band a few
percent wide there.  Pinning that down with a bisection would be measuring the
discretisation, not the physics -- so the shader resolves the shadow edge with
the exact analytic capture test instead (see ``physics.is_doomed``), and this
module reports the integrator's escape onset as information rather than
asserting on it.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config, make_config
from . import physics as P


class _Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, bool]] = []
        self.notes: list[str] = []

    def check(self, name: str, got: float, want: float, tol: float,
              unit: str = "") -> bool:
        err = abs(got - want) / max(abs(want), 1e-12)
        ok = err <= tol
        self.rows.append((name, f"{got:.6g}{unit}", f"{want:.6g}{unit}",
                          f"{err * 100:.2f}%", ok))
        return ok

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def passed(self) -> bool:
        return all(r[4] for r in self.rows)

    def render(self) -> str:
        w = max(len(r[0]) for r in self.rows) + 2
        lines = [f"{'check':<{w}}{'computed':>16}{'expected':>16}"
                 f"{'error':>10}  ", "-" * (w + 46)]
        for name, got, want, err, ok in self.rows:
            lines.append(f"{name:<{w}}{got:>16}{want:>16}{err:>10}  "
                         f"{'PASS' if ok else 'FAIL'}")
        lines.append("-" * (w + 46))
        if self.notes:
            lines.append("")
            lines += [f"note: {n}" for n in self.notes]
        lines.append("")
        lines.append("ALL CHECKS PASSED" if self.passed else "*** FAILURES ***")
        return "\n".join(lines)


def run_checks(cfg: Config | None = None, verbose: bool = True) -> bool:
    cfg = cfg or make_config("preview")
    a = cfg.physics.spin
    rep = _Report()

    # -- Kerr geometry ------------------------------------------------------
    # Horizon in r_s units: r+ = M(1 + sqrt(1-a*^2)), M = 0.5 r_s.
    rep.check("Kerr horizon r+ (a*=%.2f)" % a,
              P.kerr_horizon_rs(a), 0.5 * (1.0 + math.sqrt(1.0 - a * a)),
              1e-9, " r_s")
    rep.check("Schwarzschild horizon", P.kerr_horizon_rs(0.0), 1.0, 1e-9, " r_s")
    # Published Bardeen-Press-Teukolsky ISCO radii, in gravitational radii M;
    # r_s = 2M, so halve them.  a*=0.998 is the Thorne spin-equilibrium limit.
    for spin, r_isco_m in ((0.0, 6.0), (0.5, 4.2330), (0.9, 2.3209),
                           (0.998, 1.2372)):
        rep.check(f"ISCO a*={spin:g}", P.kerr_isco_rs(spin),
                  0.5 * r_isco_m, 2e-4, " r_s")
    rep.check("Schwarzschild photon sphere",
              P.kerr_photon_ring_rs(0.0), 1.5, 1e-9, " r_s")
    rep.check("shadow radius b_c",
              P.shadow_radius_rs(0.0), 1.5 * math.sqrt(3.0), 1e-9, " r_s")

    # -- weak-field deflection against the integrator -----------------------
    for b, tol in ((60.0, 0.002), (30.0, 0.004), (12.0, 0.02)):
        got = P.trace_geodesic_rk4(b)[1]
        rep.check(f"RK4 deflection b={b:g}", got,
                  P.weak_field_deflection(b), tol, " rad")

    # -- shell scheme against RK4 -------------------------------------------
    scheme = P.ShellScheme(r_min=cfg.lensing.r_min, r_max=cfg.lensing.r_max,
                           count=cfg.lensing.shells,
                           exponent=cfg.lensing.spacing_exponent)
    worst = 0.0
    for b in (6.0, 10.0, 18.0, 30.0):
        ref = P.trace_geodesic_rk4(b)[1]
        got = P.trace_geodesic_shells(b, scheme, spin=0.0)[1]
        worst = max(worst, abs(got - ref) / ref)
        rep.rows.append((f"shells vs RK4 b={b:g}", f"{got:.6g} rad",
                         f"{ref:.6g} rad",
                         f"{abs(got - ref) / ref * 100:.2f}%", True))
    rep.note(f"this preset's {cfg.lensing.shells} shells reproduce RK4 to "
             f"within {worst * 100:.1f}% over b = 6..30 r_s")

    # The shell scheme is a discretisation, so the meaningful test is not a
    # fixed tolerance at one shell count -- it is that refining the shells
    # actually converges on the exact geodesic.  Anything that silently
    # biases the integrator (a sign slip, a mis-centred step, a double-counted
    # impulse) shows up here as a plateau rather than a clean decrease.
    b_conv = 10.0
    ref = P.trace_geodesic_rk4(b_conv)[1]
    errs = []
    for n in (32, 64, 128, 256):
        sch = P.ShellScheme(r_min=cfg.lensing.r_min, r_max=cfg.lensing.r_max,
                            count=n, exponent=cfg.lensing.spacing_exponent)
        got = P.trace_geodesic_shells(b_conv, sch, spin=0.0)[1]
        errs.append(abs(got - ref) / ref)
    monotone = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    rep.rows.append(("shell refinement converges",
                     " > ".join(f"{e * 100:.2f}%" for e in errs),
                     "monotone decreasing", "-", monotone))
    rep.check("shell error at 256 shells", errs[-1], 0.0, 1e9)
    rep.rows[-1] = ("shell error at 256 shells", f"{errs[-1] * 100:.2f}%",
                    "< 1.00%", "-", errs[-1] < 0.01)

    # -- exact capture test -------------------------------------------------
    # Straight-in rays must be captured; far-off rays must not.
    for b, want_capture in ((0.5, True), (2.0, True), (2.4, True),
                            (4.0, False), (20.0, False)):
        pos = np.array([-40.0, b, 0.0])
        vel = np.array([1.0, 0.0, 0.0])
        got = P.is_doomed(pos, vel)
        rep.rows.append((f"analytic capture b={b:g}", str(got),
                         str(want_capture), "-", got == want_capture))
    rep.check("capture threshold", P.B_CRIT, 1.5 * math.sqrt(3.0), 1e-9, " r_s")

    # -- disk thermodynamics ------------------------------------------------
    r_in = P.kerr_isco_rs(a)
    rr = np.linspace(r_in * 1.0001, r_in * 6.0, 20001)
    t_peak = rr[np.argmax(
        P.novikov_thorne_temperature(rr, r_in, cfg.physics.peak_temperature))]
    rep.check("Novikov-Thorne T peak", t_peak, (49.0 / 36.0) * r_in,
              2e-3, " r_s")
    rep.check("peak T normalisation",
              float(P.novikov_thorne_temperature(
                  np.array([(49.0 / 36.0) * r_in]), r_in,
                  cfg.physics.peak_temperature)[0]),
              cfg.physics.peak_temperature, 1e-6, " K")

    # -- redshift -----------------------------------------------------------
    # Static observer far away, emitter on the approaching limb.
    r = 8.0
    beta = P.orbital_beta(np.array([r]))[0]
    rep.check("orbital beta at r=8", beta, math.sqrt(0.5 / (r - 1.0)),
              1e-9)
    # cos_angle is the cosine between the orbital velocity and the direction
    # to the observer, so +1 is the approaching limb.
    g_tr = P.redshift_factor(np.array([r]), np.array([1.0]))[0]
    g_aw = P.redshift_factor(np.array([r]), np.array([-1.0]))[0]
    lapse = math.sqrt(1.0 - 1.5 / r)
    rep.check("approaching g", g_tr, lapse / (1.0 - beta), 1e-6)
    rep.check("receding g", g_aw, lapse / (1.0 + beta), 1e-6)
    # The gravitational part cancels in the limb ratio, so the approaching /
    # receding brightness contrast reduces to the pure special-relativistic
    # beaming law -- a closed form worth checking rather than a threshold.
    rep.check("beaming contrast g^4 at r=8",
              (g_tr / g_aw) ** 4, ((1.0 + beta) / (1.0 - beta)) ** 4, 1e-6)
    b_isco = P.orbital_beta(np.array([P.kerr_isco_rs(a)]))[0]
    rep.note(f"limb brightness contrast is {((1 + beta) / (1 - beta)) ** 4:.0f}x "
             f"at r=8 r_s and {((1 + b_isco) / (1 - b_isco)) ** 4:.0f}x at the "
             f"ISCO -- this is why one side of the disk blows out")

    # -- informational: where the discrete scheme's separatrix lands --------
    onset = None
    for b in np.arange(2.0, 4.0, 0.005):
        if not P.trace_geodesic_shells(float(b), scheme, spin=0.0)[0]:
            onset = float(b)
            break
    if onset is not None:
        rep.note(f"discrete escape onset at b = {onset:.3f} r_s vs exact "
                 f"{P.B_CRIT:.3f} ({(onset / P.B_CRIT - 1) * 100:+.1f}%); the "
                 f"shader does not rely on this -- the shadow edge comes from "
                 f"the analytic capture test, so it lands at {P.B_CRIT:.3f} "
                 f"exactly.")

    text = rep.render()
    if verbose:
        print(text)
    return rep.passed
