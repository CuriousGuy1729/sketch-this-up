"""
Stage 1 + 5 -- cinematic camera rig and animation.

Rig: an Empty at the black hole's centre, with the camera parented to it and
constrained to track it.  Animating the empty's Z rotation orbits the camera;
animating the camera's local position dollies it.  Keeping the two concerns on
separate objects means the orbit and the dolly can have independent easing,
which is what stops the move feeling mechanical.

The shot is framed near edge-on (~84 deg from the disk axis) because that is
where relativistic lensing is most legible: the far side of the disk is lifted
into an arc above the shadow and folded back underneath it, and the Doppler
asymmetry across the disk is at its maximum.
"""

from __future__ import annotations

import math

import bpy

from .config import Config


def _keyframe(obj, data_path: str, index: int, frames, values,
              interpolation: str = "BEZIER", easing: str = "EASE_IN_OUT"):
    for frame, value in zip(frames, values):
        if index < 0:
            setattr(obj, data_path, value)
        else:
            getattr(obj, data_path)[index] = value
        obj.keyframe_insert(data_path=data_path,
                            index=index, frame=frame)
    action = obj.animation_data.action
    # Blender 4.4+ organises actions into slots/layers; fall back gracefully.
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:                                    # pragma: no cover
        return
    for fc in fcurves:
        if fc.data_path != data_path or (index >= 0 and fc.array_index != index):
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation
            kp.easing = easing


def build_camera(cfg: Config) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Create the rig and animate it.  Returns ``(camera, pivot)``."""
    cam_cfg = cfg.camera
    scene = bpy.context.scene
    last = cfg.frame_count

    pivot = bpy.data.objects.new("BH_CameraPivot", None)
    pivot.empty_display_size = 2.0
    bpy.context.collection.objects.link(pivot)

    cam_data = bpy.data.cameras.new("BH_CameraData")
    cam_data.lens = cam_cfg.focal_length
    cam_data.sensor_width = cam_cfg.sensor_width
    cam_data.clip_start = 0.01
    cam_data.clip_end = 20000.0

    cam = bpy.data.objects.new("BH_Camera", cam_data)
    cam.parent = pivot
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    track = cam.constraints.new("TRACK_TO")
    track.target = pivot
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # --- dolly: position in the pivot's local frame ------------------------
    def place(distance: float, inclination_deg: float) -> tuple[float, float, float]:
        # Inclination measured from the spin axis: 90 deg = edge-on.
        theta = math.radians(inclination_deg)
        return (0.0, -distance * math.sin(theta), distance * math.cos(theta))

    p0 = place(cam_cfg.distance_start, cam_cfg.inclination_start)
    p1 = place(cam_cfg.distance_end, cam_cfg.inclination_end)
    for axis in range(3):
        _keyframe(cam, "location", axis, (1, last), (p0[axis], p1[axis]))

    # --- orbit: pivot spin -------------------------------------------------
    _keyframe(pivot, "rotation_euler", 2,
              (1, last),
              (math.radians(cam_cfg.azimuth_start),
               math.radians(cam_cfg.azimuth_end)),
              easing="EASE_IN_OUT")

    # --- roll --------------------------------------------------------------
    track.influence = 1.0
    _keyframe(cam, "rotation_euler", 2,
              (1, last),
              (math.radians(cam_cfg.roll_start),
               math.radians(cam_cfg.roll_end)))

    # --- depth of field ----------------------------------------------------
    if cam_cfg.use_dof:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = pivot
        cam_data.dof.aperture_fstop = cam_cfg.f_stop
        cam_data.dof.aperture_blades = 7

    return cam, pivot
