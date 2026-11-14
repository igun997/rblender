"""Fan-made procedural 3D model + rig + animation of Vestia Zeta (hololive Indonesia), built entirely
with bpy on a headless remote Blender. Not an official asset; character (c) COVER Corp.

    rblender run characters/vestia_zeta.py -- [--quick] [--no-video] [--no-render]
"""
import json
import time

import bpy
import rb

from characters.zeta import anim, body, hair, mats, outfit, stage
from characters.zeta import rig as rigmod
from characters.zeta import skeleton as SK
from characters.zeta.common import OBJECTS

opts = rb.args()
QUICK = "--quick" in opts
NO_VIDEO = "--no-video" in opts or QUICK
NO_RENDER = "--no-render" in opts
T0 = time.time()
timing = {}


def lap(name):
    timing[name] = round(time.time() - T0, 1)


# ------------------------------------------------------------------ model
scene = rb.clean()
stage.look(scene)
M = mats.build()
SK.build_base()
body.build(M)
hair.build(M)
outfit.build(M)
lap("model")

char_objects = [o for o in OBJECTS.values() if o.name in bpy.data.objects]
verts = sum(len(o.data.vertices) for o in char_objects)
tris = sum(len(p.vertices) - 2 for o in char_objects for p in o.data.polygons)

# ------------------------------------------------------------------ rig + animation
rig = rigmod.build_armature()
bound, unweighted = rigmod.bind(rig)
lap("rig")
fk_err, hand_to_brow = anim.verify_against_blender(rig)
anim_info = anim.bake(rig)
lap("anim")

# ------------------------------------------------------------------ studio (after the asset is complete)
stage.world()
stage.floor()
stage.lights()


def select_asset():
    bpy.ops.object.select_all(action="DESELECT")
    for o in char_objects + [rig]:
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig


scene.frame_set(1)
select_asset()
export_opts = dict(use_selection=True)
rb.export("vestia_zeta.glb", export_skins=True, **export_opts)
rb.export("vestia_zeta.fbx", add_leaf_bones=False, **export_opts)
lap("export")

# ------------------------------------------------------------------ renders
cams = {
    "hero": stage.orbit(-22, dist=3.9, height=1.05, target=(0, 0, 0.85), lens=62, name="CamHero"),
    "front": stage.orbit(0, dist=3.7, height=1.0, target=(0, 0, 0.82), lens=60, name="CamFront"),
    "face": stage.camera((0.55, -1.25, 1.50), (0.02, 0, 1.43), lens=85, name="CamFace"),
}
SALUTE_FRAME = int(2.2 * anim.FPS) + 1
scale = 0.5 if QUICK else 1.0
if not NO_RENDER:
    scene.camera = cams["hero"]
    rb.render_still("zeta_hero.png", engine="CYCLES", samples=24 if QUICK else 64, res=(int(810 * scale), int(1440 * scale)), frame=SALUTE_FRAME)
    lap("hero")
    scene.camera = cams["face"]
    rb.render_still("zeta_face.png", engine="EEVEE", samples=48, res=(int(900 * scale), int(900 * scale)), frame=SALUTE_FRAME)
    lap("face")
    # turnaround at the idle pose
    files = []
    for ang in (0, 60, 120, 180, 240, 300):
        scene.camera = stage.orbit(ang, dist=3.7, height=1.0, target=(0, 0, 0.82), lens=60, name=f"CamTurn{ang}")
        files.append(rb.render_still(f"_turn_{ang}.png", engine="EEVEE", samples=32, res=(int(360 * scale), int(640 * scale)), frame=1))
    stage.contact_sheet(files, rb.out("zeta_turnaround.png"))
    # animation strip
    files = []
    scene.camera = cams["front"]
    for sec in (0.0, 1.2, 2.2, 3.3, 3.95, 4.8):
        files.append(rb.render_still(f"_strip_{sec}.png", engine="EEVEE", samples=32, res=(int(360 * scale), int(640 * scale)), frame=int(sec * anim.FPS) + 1))
    stage.contact_sheet(files, rb.out("zeta_motion_strip.png"))
    lap("stills")
    # topology
    scene.camera = cams["front"]
    stage.wireframe_render(rb.out("zeta_wireframe.png"), res=(int(540 * scale), int(960 * scale)), samples=8)
    lap("wire")
    if not NO_VIDEO:
        scene.camera = cams["hero"]
        rb.render_anim("zeta_mission_start.mp4", engine="EEVEE", samples=16, res=(450, 800), frames=(1, anim.int_frames()))
        lap("video")

import os
for fn in os.listdir(rb.OUT_DIR):  # intermediate frames are not deliverables
    if fn.startswith("_"):
        os.remove(os.path.join(rb.OUT_DIR, fn))

info = {
    "objects": len(char_objects),
    "vertices": verts,
    "triangles": tris,
    "materials": len(bpy.data.materials),
    "images": len(bpy.data.images),
    "bones": len(rig.data.bones),
    "meshes_bound": bound,
    "unweighted_vertices": unweighted,
    "fk_vs_blender_max_error_m": round(fk_err, 6),
    "salute_hand_to_brow_target_m": round(hand_to_brow, 4),
    "animation": anim_info,
    "seconds": timing,
}
with open(rb.out("zeta_stats.json"), "w") as fh:
    json.dump(info, fh, indent=2)
rb.report(zeta=info)
