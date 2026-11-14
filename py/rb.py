"""rblender helper module. Available to every script run by `rblender run|eval|demo`.

    import rb
    rb.clean()                         # empty scene, no default cube/light/camera
    rb.out("shot.png")                 # absolute path inside the output dir (pulled back to ./out)
    rb.args()                          # argv after `--`
    rb.render_still("a.png", engine="CYCLES", samples=16, res=(640, 360))
    rb.render_anim("a.mp4", frames=(1, 48))
    rb.export("model.glb")             # by extension: glb gltf fbx obj usd usdc usda stl ply abc blend
"""
import json
import os
import sys

import bpy

OUT_DIR = os.environ.get("RBLENDER_OUT") or os.path.join(os.getcwd(), "out")
os.makedirs(OUT_DIR, exist_ok=True)

_written = []


def args():
    """Arguments after the `--` separator on the blender command line."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def out(name):
    """Absolute path for an output artifact; it is pulled back to the local ./out afterwards."""
    path = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path not in _written:
        _written.append(path)
    return path


def clean():
    """Remove everything (objects, materials, images, ...) for a reproducible scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _setup_render(scene, engine, samples, res):
    engines = {
        "CYCLES": "CYCLES",
        "EEVEE": "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2) else "BLENDER_EEVEE",
        "WORKBENCH": "BLENDER_WORKBENCH",
    }
    scene.render.engine = engines.get(engine.upper(), engine)
    if scene.render.engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False
    elif hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = samples
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100


def render_still(name, engine="CYCLES", samples=16, res=(640, 360), frame=None):
    scene = bpy.context.scene
    _setup_render(scene, engine, samples, res)
    if frame is not None:
        scene.frame_set(frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out(name)
    bpy.ops.render.render(write_still=True)
    return scene.render.filepath


def render_anim(name, frames=None, engine="CYCLES", samples=8, res=(640, 360), fps=24):
    """name ending in .mp4 -> H.264 video; otherwise a PNG sequence `<name>_0001.png`."""
    scene = bpy.context.scene
    _setup_render(scene, engine, samples, res)
    scene.render.fps = fps
    if frames:
        scene.frame_start, scene.frame_end = frames
    if name.lower().endswith(".mp4"):
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.filepath = out(name)
        final = scene.render.filepath
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = out(name + "_")
        final = scene.render.filepath
    bpy.ops.render.render(animation=True)
    return final


def export(name, **opts):
    """Export the current scene by file extension. Extra kwargs go to the blender operator."""
    path = out(name)
    ext = name.rsplit(".", 1)[-1].lower()
    ops = bpy.ops
    if ext in ("glb", "gltf"):
        ops.export_scene.gltf(filepath=path, export_format="GLB" if ext == "glb" else "GLTF_SEPARATE",
                              export_animations=True, **opts)
    elif ext == "fbx":
        ops.export_scene.fbx(filepath=path, bake_anim=True, **opts)
    elif ext == "obj":
        ops.wm.obj_export(filepath=path, **opts)
    elif ext in ("usd", "usdc", "usda"):
        ops.wm.usd_export(filepath=path, export_animation=True, **opts)
    elif ext == "stl":
        ops.wm.stl_export(filepath=path, **opts)
    elif ext == "ply":
        ops.wm.ply_export(filepath=path, **opts)
    elif ext == "abc":
        ops.wm.alembic_export(filepath=path, start=bpy.context.scene.frame_start,
                              end=bpy.context.scene.frame_end, **opts)
    elif ext == "blend":
        ops.wm.save_as_mainfile(filepath=path, **opts)
    else:
        raise ValueError(f"unsupported export extension: .{ext}")
    return path


def report(**extra):
    """Write report.json (blender version + every artifact written via rb.out) and print it."""
    data = {
        "blender": bpy.app.version_string,
        "engine": bpy.context.scene.render.engine,
        "files": {os.path.basename(p): os.path.getsize(p) for p in _written if os.path.exists(p)},
    }
    data.update(extra)
    with open(os.path.join(OUT_DIR, "report.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print("RBLENDER_REPORT " + json.dumps(data))
    return data
