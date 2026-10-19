"""
Stages 6 and 7 -- rendering the frame sequence and encoding it to H.264.

Frames are written to disk as individual 16-bit PNGs and only then muxed with
FFmpeg.  That is deliberate:

* a Cycles render of this scene is long enough that losing it to a crash in
  hour nine would be unacceptable, so the loop is **resumable** -- re-running
  skips frames that already exist;
* 16-bit PNG keeps the full tonal range of the accretion disk through to the
  encoder, so the gradient across the photon ring does not band;
* encoding is a separate step, which means the video can be re-graded or
  re-encoded at a different bitrate without re-rendering anything.

FFmpeg is located via :mod:`imageio_ffmpeg`, which ships a static build, so
there is no dependency on a system FFmpeg install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import bpy

from .config import Config


# ---------------------------------------------------------------------------
# FFmpeg discovery
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    """
    Return a usable ffmpeg executable.

    Prefers the wheel-provided static build (always present once
    ``requirements.txt`` is installed, and always has libx264), then falls
    back to whatever is on ``PATH``.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    raise RuntimeError(
        "No ffmpeg binary found. Install it with 'pip install imageio-ffmpeg' "
        "or put ffmpeg on your PATH.")


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def frame_path(cfg: Config, frame: int) -> str:
    return os.path.join(cfg.frames_dir, f"frame_{frame:04d}.png")


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def render_animation(cfg: Config, start: int | None = None,
                     end: int | None = None, overwrite: bool = False) -> list[str]:
    """
    Render the frame range to ``cfg.frames_dir`` and return the file list.

    Rendering one frame at a time with :func:`bpy.ops.render.render` rather
    than ``animation=True`` costs nothing (persistent data keeps the BVH and
    shaders resident between frames) and buys the resume behaviour, a live ETA,
    and the ability to interrupt cleanly.
    """
    scene = bpy.context.scene
    start = scene.frame_start if start is None else start
    end = scene.frame_end if end is None else end

    os.makedirs(cfg.frames_dir, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 15

    total = end - start + 1
    produced: list[str] = []
    times: list[float] = []

    print(f"[render] {total} frames at {cfg.render.resolution_x}x"
          f"{cfg.render.resolution_y}, {cfg.render.samples} spp "
          f"-> {cfg.frames_dir}")

    for i, frame in enumerate(range(start, end + 1)):
        path = frame_path(cfg, frame)
        if os.path.exists(path) and not overwrite:
            produced.append(path)
            print(f"[render] {frame:4d}/{end}  skip (exists)")
            continue

        scene.frame_set(frame)
        scene.render.filepath = path

        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        times.append(dt)

        mean = sum(times) / len(times)
        eta = mean * (total - i - 1)
        print(f"[render] {frame:4d}/{end}  {dt:6.1f}s  "
              f"avg {mean:6.1f}s  eta {_fmt_hms(eta)}", flush=True)
        produced.append(path)

    if times:
        print(f"[render] done, {sum(times) / 60.0:.1f} min of render time")
    return produced


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_video(cfg: Config, output: str | None = None,
                 crf: int = 14, preset: str = "slow") -> str:
    """
    Mux the PNG sequence into an H.264 MP4.

    * ``yuv420p`` and ``profile high`` for universal playback -- Cycles writes
      RGB, and without the explicit pixel format FFmpeg would pick yuv444p,
      which many players and browsers refuse.
    * ``-crf 14`` is visually lossless for this material; the smooth gradients
      across the disk are exactly what shows compression artefacts first.
    * ``+faststart`` relocates the moov atom so the file streams without a
      full download.
    * ``-color_primaries/-colorspace bt709`` tags the stream so players do not
      guess the matrix and shift the colour of the disk.
    """
    output = output or cfg.video_path
    ffmpeg = find_ffmpeg()

    first = frame_path(cfg, bpy.context.scene.frame_start) \
        if bpy.context.scene else None
    pattern = os.path.join(cfg.frames_dir, "frame_%04d.png")

    existing = sorted(f for f in os.listdir(cfg.frames_dir)
                      if f.startswith("frame_") and f.endswith(".png"))
    if not existing:
        raise RuntimeError(f"No frames to encode in {cfg.frames_dir}")
    start_number = int(existing[0][6:10])

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(cfg.render.fps),
        "-start_number", str(start_number),
        "-i", pattern,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "5.2",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
        output,
    ]

    print(f"[encode] {len(existing)} frames @ {cfg.render.fps} fps -> {output}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    size_mb = os.path.getsize(output) / 1e6
    print(f"[encode] wrote {output}  ({size_mb:.1f} MB)")
    return output


def save_blend(cfg: Config, path: str | None = None) -> str:
    """Save the fully built scene so it can be opened in the Blender GUI."""
    path = os.path.abspath(path or cfg.blend_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    print(f"[blend] saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path
