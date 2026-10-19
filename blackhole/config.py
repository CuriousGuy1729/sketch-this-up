"""
Central configuration: physical parameters, look development, quality presets.

Everything the renderer needs is in one dataclass tree so a run is fully
described by a single ``Config`` object (and can be dumped next to the render
for reproducibility).

Scene units: **1 Blender unit = 1 Schwarzschild radius** (see physics.py).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict


# =============================================================================

@dataclass
class PhysicsConfig:
    """The black hole and its accretion flow."""

    #: Dimensionless Kerr spin a* = Jc/GM^2, in [0, 1).  0.9 is typical of a
    #: well-fed supermassive hole and gives a visibly compact ISCO.
    spin: float = 0.90

    #: Outer edge of the modelled disk, in Schwarzschild radii.
    disk_outer: float = 16.0

    #: Inner edge.  ``None`` means "use the Kerr ISCO for this spin".
    disk_inner: float | None = None

    #: Geometric half-thickness of the disk at r = 10 r_s.  Real thin disks
    #: have H/R ~ 0.01-0.1; we exaggerate slightly so the flow reads as a
    #: volume rather than a sheet of paper.
    disk_thickness: float = 0.20

    #: Flaring exponent: H(r) ~ (r/10)^flare.  9/8 is the Shakura-Sunyaev
    #: outer-region value.
    disk_flare: float = 1.05

    #: Peak colour temperature of the disk in kelvin.  The *physical* value for
    #: a supermassive hole is ~1e5 K (far UV -> featureless blue-white), so the
    #: cinematic default remaps it into the visible.  Use
    #: ``--physical-temperature`` to compute it from mass + accretion rate.
    peak_temperature: float = 9200.0

    #: Only used by ``--physical-temperature``.
    mass_solar: float = 1.0e7
    eddington_ratio: float = 0.10

    #: Exponent in the relativistic beaming law I_obs = g^beaming * I_emit.
    #: 4 is the bolometric result; lower it to tame the contrast artistically.
    beaming_exponent: float = 4.0

    #: Overall disk emission gain (pre-tonemap).
    disk_brightness: float = 22.0

    #: Opacity of the plasma -- controls how much of the far side shows through.
    disk_opacity: float = 0.72


@dataclass
class LensingConfig:
    """The nested ray-portal shells that integrate null geodesics."""

    #: Number of concentric shells = number of integration steps per radial
    #: pass.  This is the single biggest quality/cost dial.
    shells: int = 128

    #: Innermost and outermost shell radius (r_s).
    r_min: float = 1.05
    r_max: float = 45.0

    #: Shell spacing law: 0 = logarithmic (dr/r constant).  Negative values
    #: concentrate shells near the hole.  Calibration found logarithmic to be
    #: the best compromise -- see ``validate.py``.
    spacing_exponent: float = 0.0

    #: Subdivision level of the icosphere used for each shell.  The shader
    #: reads the true |P| so faceting is self-consistent, but low values make
    #: the *radii* of crossings lumpy.
    sphere_subdivisions: int = 5

    #: Enable the Lense-Thirring (frame-dragging) term.
    frame_dragging: bool = True

    #: Multiplies every deflection impulse.  1.0 = general relativity.
    #: Exposed for A/B comparisons, not for cheating.
    deflection_gain: float = 1.0


@dataclass
class StarfieldConfig:
    sky_strength: float = 1.0
    star_density: float = 1.0
    star_brightness: float = 1.6
    #: Equirectangular resolution of the NumPy-generated nebula/Milky Way map.
    nebula_width: int = 4096
    nebula_strength: float = 0.26
    seed: int = 20251019


@dataclass
class CameraConfig:
    """Cinematic camera rig: an empty at the origin with the camera parented."""

    focal_length: float = 70.0
    sensor_width: float = 36.0

    #: Distance from the hole, in r_s, at the start and end of the shot.
    #: MUST stay outside ``LensingConfig.r_max`` -- a camera inside the shell
    #: stack starts its rays mid-integration and the shells themselves become
    #: visible as concentric ripples.  ``validate_config`` enforces this.
    distance_start: float = 72.0
    distance_end: float = 56.0

    #: Inclination measured from the disk plane (90 deg = edge-on).  Near
    #: edge-on is where lensing is most dramatic -- the far side of the disk
    #: arcs up over the shadow.
    inclination_start: float = 84.0
    inclination_end: float = 72.0

    #: Azimuthal travel over the shot, degrees.
    azimuth_start: float = -14.0
    azimuth_end: float = 16.0

    #: Slight roll for a less "CG turntable" feel.
    roll_start: float = -1.5
    roll_end: float = 1.0

    use_dof: bool = True
    f_stop: float = 6.0


@dataclass
class RenderConfig:
    resolution_x: int = 3840
    resolution_y: int = 2160
    fps: int = 60
    duration_seconds: float = 14.0

    samples: int = 1024
    adaptive_threshold: float = 0.01
    max_bounces: int = 8
    #: Ray-portal crossings consume *transparent* bounces: this must exceed
    #: the number of shells a ray can traverse, with headroom for photons that
    #: wind several times around the photon sphere.
    transparent_bounces: int = 384

    denoise: bool = True
    use_persistent_data: bool = True
    #: 'AgX' handles the huge dynamic range of g^4 beaming far better than
    #: 'Filmic' or 'Standard'.
    view_transform: str = "AgX"
    look: str = "AgX - Medium High Contrast"
    exposure: float = 0.0

    glare_strength: float = 0.08
    glare_size: int = 8
    vignette: float = 0.22


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    lensing: LensingConfig = field(default_factory=LensingConfig)
    stars: StarfieldConfig = field(default_factory=StarfieldConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    quality: str = "final"

    # -- output locations ---------------------------------------------------
    output_dir: str = "output"
    video_name: str = "black_hole_4k.mp4"
    blend_name: str = "black_hole.blend"

    # -- derived ------------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return max(1, round(self.render.duration_seconds * self.render.fps))

    @property
    def frames_dir(self) -> str:
        """Frames are kept per-quality so a preview never clobbers a 4K run."""
        return os.path.join(self.output_dir, f"frames_{self.quality}")

    @property
    def video_path(self) -> str:
        if self.quality == "final":
            return os.path.join(self.output_dir, self.video_name)
        stem, ext = os.path.splitext(self.video_name)
        return os.path.join(self.output_dir, f"{stem}_{self.quality}{ext}")

    @property
    def blend_path(self) -> str:
        return os.path.join(self.output_dir, self.blend_name)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def describe(self) -> str:
        r = self.render
        return (f"{r.resolution_x}x{r.resolution_y} @ {r.fps}fps | "
                f"{self.frame_count} frames ({r.duration_seconds:g}s) | "
                f"{r.samples} spp | {self.lensing.shells} lensing shells")


# =============================================================================
# Quality presets
# =============================================================================
#
# ``preview`` is sized so a single frame finishes in seconds on a laptop CPU;
# ``final`` is the 4K deliverable.  Everything in between scales the three
# things that actually dominate cost: pixels, samples, and shell count.
# =============================================================================

PRESETS: dict[str, dict] = {
    "thumb": dict(                      # smoke test: is the scene even sane?
        resolution_x=480, resolution_y=270, samples=24, shells=40,
        transparent_bounces=110, sphere_subdivisions=4, denoise=False,
        nebula_width=1024, duration_seconds=14.0,
    ),
    "preview": dict(                    # fast motion/lighting check
        resolution_x=960, resolution_y=540, samples=64, shells=64,
        transparent_bounces=180, sphere_subdivisions=4, denoise=True,
        nebula_width=2048, duration_seconds=14.0,
    ),
    "high": dict(                        # 1080p near-final look
        resolution_x=1920, resolution_y=1080, samples=384, shells=96,
        transparent_bounces=280, sphere_subdivisions=5, denoise=True,
        nebula_width=4096, duration_seconds=14.0,
    ),
    "final": dict(                       # 4K UHD deliverable
        resolution_x=3840, resolution_y=2160, samples=1024, shells=128,
        transparent_bounces=384, sphere_subdivisions=5, denoise=True,
        nebula_width=8192, duration_seconds=14.0,
    ),
}


def validate_config(cfg: Config) -> None:
    """Catch the geometry mistakes that silently ruin a render."""
    lens, cam, phys = cfg.lensing, cfg.camera, cfg.physics
    nearest = min(cam.distance_start, cam.distance_end)
    if nearest <= lens.r_max * 1.02:
        raise SystemExit(
            f"camera distance ({nearest:g} r_s) must stay outside the lensing "
            f"shell stack (r_max = {lens.r_max:g} r_s). Either move the camera "
            f"back or lower lensing.r_max.")
    if phys.disk_outer >= lens.r_max:
        raise SystemExit(
            f"disk_outer ({phys.disk_outer:g}) must be inside r_max "
            f"({lens.r_max:g}) or the disk will not be lensed.")
    need = 2 * lens.shells
    if cfg.render.transparent_bounces < need:
        raise SystemExit(
            f"transparent_bounces ({cfg.render.transparent_bounces}) must be "
            f"at least 2 x shells ({need}) or rays die mid-integration.")


def make_config(quality: str = "final", **overrides) -> Config:
    """Build a ``Config`` from a preset name plus ad-hoc overrides."""
    if quality not in PRESETS:
        raise SystemExit(f"unknown quality '{quality}'; "
                         f"choose from {', '.join(PRESETS)}")
    cfg = Config(quality=quality)
    preset = dict(PRESETS[quality])
    preset.update({k: v for k, v in overrides.items() if v is not None})

    groups = (cfg.render, cfg.lensing, cfg.stars, cfg.camera, cfg.physics)
    for key, value in preset.items():
        for group in groups:
            if hasattr(group, key):
                setattr(group, key, value)
                break
        else:
            raise SystemExit(f"unknown config key '{key}'")
    validate_config(cfg)
    return cfg
