#!/usr/bin/env python3
"""
Cinematic Kerr black hole -- command line entry point.

    python main.py --preview              fast single frame, look check
    python main.py --render               render the full frame sequence
    python main.py --render-and-encode    render, then mux to H.264 MP4

Runs against the ``bpy`` PyPI module (no Blender install required); it also
works when executed inside Blender as ``blender -b -P main.py -- --preview``.

The seven build stages live in separate modules and run in order from
``scene.build_scene``:

    1. black hole + camera        physics.py, camera.py
    2. accretion disk             disk.py
    3. lensing + relativity       lensing.py, nodes.py
    4. stars + lighting           starfield.py
    5. animation                  camera.py, disk.py (keyframes/drivers)
    6. render                     render.py
    7. encode                     render.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _argv() -> list[str]:
    """Support both ``python main.py ...`` and ``blender -b -P main.py -- ...``."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return sys.argv[1:]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Render a physically-derived Kerr black hole.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    mode = p.add_argument_group("mode (pick one)")
    mode.add_argument("--preview", action="store_true",
                      help="render one frame at preview quality and stop")
    mode.add_argument("--render", action="store_true",
                      help="render the full frame sequence")
    mode.add_argument("--render-and-encode", dest="render_and_encode",
                      action="store_true",
                      help="render the sequence, then encode the MP4")
    mode.add_argument("--encode-only", action="store_true",
                      help="encode already-rendered frames and stop")
    mode.add_argument("--validate", action="store_true",
                      help="run the physics self-checks (no Blender needed)")
    mode.add_argument("--build-only", dest="build_only", action="store_true",
                      help="build the scene and save the .blend, render nothing")

    opt = p.add_argument_group("options")
    opt.add_argument("-q", "--quality",
                     choices=("thumb", "preview", "high", "final"),
                     default=None,
                     help="quality preset (default: preview for --preview, "
                          "final otherwise)")
    opt.add_argument("--frame", type=int, default=None,
                     help="single frame to render in --preview mode")
    opt.add_argument("--start", type=int, default=None, help="first frame")
    opt.add_argument("--end", type=int, default=None, help="last frame")
    opt.add_argument("--resolution", type=int, nargs=2,
                     metavar=("W", "H"), default=None,
                     help="override the resolution")
    opt.add_argument("--samples", type=int, default=None,
                     help="override samples per pixel")
    opt.add_argument("--shells", type=int, default=None,
                     help="override the number of lensing shells")
    opt.add_argument("--seconds", type=float, default=None,
                     dest="duration_seconds",
                     help="override the shot length in seconds")
    opt.add_argument("--spin", type=float, default=None,
                     help="dimensionless Kerr spin a* in [0, 0.999]")
    opt.add_argument("--nebula-width", type=int, default=None,
                     dest="nebula_width",
                     help="equirect width of the synthesised nebula; lower it "
                          "if the 8K default is too heavy for your RAM")
    opt.add_argument("--output-dir", default=None,
                     help="where frames, the .blend and the MP4 go")
    opt.add_argument("--no-blend", action="store_true",
                     help="skip saving the .blend file")
    opt.add_argument("--overwrite", action="store_true",
                     help="re-render frames that already exist")
    opt.add_argument("--crf", type=int, default=14,
                     help="x264 quality, lower is better (default 14)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_argv() if argv is None else argv)

    modes = (args.preview, args.render, args.render_and_encode,
             args.encode_only, args.validate, args.build_only)
    if sum(bool(m) for m in modes) != 1:
        build_parser().print_help()
        print("\nerror: pick exactly one of --preview, --render, "
              "--render-and-encode, --encode-only, --validate, --build-only",
              file=sys.stderr)
        return 2

    # Make sure the package is importable when run from anywhere.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    from blackhole.config import make_config

    quality = args.quality or ("preview" if args.preview else "final")
    overrides = {}
    if args.resolution:
        overrides["resolution_x"], overrides["resolution_y"] = args.resolution
    for key in ("samples", "shells", "duration_seconds", "spin",
                "nebula_width"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)

    # Every shell crossing spends a transparent bounce, so the budget has to
    # scale with --shells or rays die mid-integration.  Do it before
    # make_config, which validates the two against each other.
    if args.shells is not None:
        from blackhole.config import PRESETS
        overrides["transparent_bounces"] = max(
            PRESETS[quality]["transparent_bounces"], 3 * args.shells)

    cfg = make_config(quality, **overrides)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    # ---------------------------------------------------------------- validate
    if args.validate:
        from blackhole.validate import run_checks
        return 0 if run_checks(cfg) else 1

    print(f"[config] {quality}: {cfg.describe()}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    from blackhole.render import encode_video, render_animation, save_blend

    # ------------------------------------------------------------- encode only
    if args.encode_only:
        encode_video(cfg, crf=args.crf)
        return 0

    # -------------------------------------------------------------- build scene
    t0 = time.time()
    from blackhole.scene import build_scene

    build_scene(cfg)
    print(f"[scene]  built in {time.time() - t0:.1f}s")

    if not args.no_blend:
        save_blend(cfg)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as fh:
        fh.write(cfg.to_json())

    if args.build_only:
        print("[build] scene built and saved; nothing rendered")
        return 0

    # ------------------------------------------------------------------ preview
    if args.preview:
        import bpy

        frame = args.frame if args.frame is not None \
            else bpy.context.scene.frame_start
        os.makedirs(cfg.frames_dir, exist_ok=True)
        still = os.path.join(cfg.output_dir, f"preview_{frame:04d}.png")

        bpy.context.scene.frame_set(frame)
        bpy.context.scene.render.filepath = still
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[preview] frame {frame} in {time.time() - t0:.1f}s -> {still}")
        return 0

    # ------------------------------------------------------------------- render
    render_animation(cfg, start=args.start, end=args.end,
                     overwrite=args.overwrite)

    if args.render_and_encode:
        encode_video(cfg, crf=args.crf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
