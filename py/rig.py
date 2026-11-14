"""rblender PoC: model a character, rig it (armature + skin weights), pose-animate it, render, export.

Usage: rblender rig [--engine cycles|eevee] [--quick]
Exports: rig.glb (skin + animation), rig.fbx, rig.blend, rig_still.png, rig_walk.mp4, rig_bones.json
"""
import json
import math

import bpy
import rb
from mathutils import Vector

opts = rb.args()
engine = opts[opts.index("--engine") + 1] if "--engine" in opts else "CYCLES"
quick = "--quick" in opts
N_FRAMES = 12 if quick else 24

scene = rb.clean()
scene.frame_start, scene.frame_end = 1, N_FRAMES


def part(kind, name, loc, scale, mat, cuts=4):
    """A subdivided primitive; the extra loops give the skin something to bend at."""
    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=loc)
    else:
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    if kind == "cylinder":
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=cuts)
        bpy.ops.object.mode_set(mode="OBJECT")
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def material(name, color, rough=0.5, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


skin = material("Skin", (0.8, 0.55, 0.4, 1), 0.6)
cloth = material("Cloth", (0.1, 0.3, 0.8, 1), 0.8)
pants = material("Pants", (0.15, 0.15, 0.2, 1), 0.8)

# ---- mesh: a simple humanoid made of separate segments, joined into ONE skinned mesh
parts = [
    part("cylinder", "torso", (0, 0, 1.55), (0.28, 0.16, 0.45), cloth, 6),
    part("sphere", "head", (0, 0, 2.3), (0.2, 0.2, 0.22), skin),
    part("cylinder", "upper_arm.L", (0.55, 0, 1.8), (0.08, 0.08, 0.28), cloth, 5),
    part("cylinder", "lower_arm.L", (0.55, 0, 1.25), (0.07, 0.07, 0.28), skin, 5),
    part("cylinder", "upper_arm.R", (-0.55, 0, 1.8), (0.08, 0.08, 0.28), cloth, 5),
    part("cylinder", "lower_arm.R", (-0.55, 0, 1.25), (0.07, 0.07, 0.28), skin, 5),
    part("cylinder", "thigh.L", (0.17, 0, 0.78), (0.11, 0.11, 0.3), pants, 5),
    part("cylinder", "shin.L", (0.17, 0, 0.2), (0.09, 0.09, 0.3), pants, 5),
    part("cylinder", "thigh.R", (-0.17, 0, 0.78), (0.11, 0.11, 0.3), pants, 5),
    part("cylinder", "shin.R", (-0.17, 0, 0.2), (0.09, 0.09, 0.3), pants, 5),
]
bpy.ops.object.select_all(action="DESELECT")
for p in parts:
    p.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
body = bpy.context.object
body.name = "Character"

# ---- armature
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm_obj = bpy.context.object
arm_obj.name = "Rig"
arm = arm_obj.data
arm.name = "RigArmature"
eb = arm.edit_bones
eb.remove(eb[0])

BONES = [  # name, head, tail, parent
    ("root", (0, 0, 0.95), (0, 0, 1.15), None),
    ("spine", (0, 0, 1.15), (0, 0, 1.95), "root"),
    ("head", (0, 0, 1.95), (0, 0, 2.55), "spine"),
    ("upper_arm.L", (0.55, 0, 2.05), (0.55, 0, 1.5), "spine"),
    ("lower_arm.L", (0.55, 0, 1.5), (0.55, 0, 0.95), "upper_arm.L"),
    ("upper_arm.R", (-0.55, 0, 2.05), (-0.55, 0, 1.5), "spine"),
    ("lower_arm.R", (-0.55, 0, 1.5), (-0.55, 0, 0.95), "upper_arm.R"),
    ("thigh.L", (0.17, 0, 1.08), (0.17, 0, 0.5), "root"),
    ("shin.L", (0.17, 0, 0.5), (0.17, 0, 0.0), "thigh.L"),
    ("thigh.R", (-0.17, 0, 1.08), (-0.17, 0, 0.5), "root"),
    ("shin.R", (-0.17, 0, 0.5), (-0.17, 0, 0.0), "thigh.R"),
]
for name, head, tail, parent in BONES:
    b = eb.new(name)
    b.head, b.tail = Vector(head), Vector(tail)
    if parent:
        b.parent = eb[parent]
        b.use_connect = Vector(head) == Vector(eb[parent].tail)
bpy.ops.object.mode_set(mode="OBJECT")

# ---- skinning: automatic (bone heat) weights, verified; fall back to envelope weights
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
method = "ARMATURE_AUTO"
bpy.ops.object.parent_set(type="ARMATURE_AUTO")


def unweighted(obj):
    return sum(1 for v in obj.data.vertices if not any(g.weight > 0 for g in v.groups))


bad = unweighted(body)
if bad or not body.vertex_groups:
    # heat-diffusion can fail on disjoint islands: redo with envelope weights
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    body.vertex_groups.clear()
    for m in [m for m in body.modifiers if m.type == "ARMATURE"]:
        body.modifiers.remove(m)
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    method = "ARMATURE_ENVELOPE"
    bad = unweighted(body)

# ---- animation: a walk cycle with pose bones (FK), looping
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode="POSE")
pb = arm_obj.pose.bones
for pbone in pb:
    pbone.rotation_mode = "XYZ"


def swing(bone, f, amp, phase=0.0, axis=0):
    t = (f - 1) / N_FRAMES
    ang = math.sin(t * math.tau + phase) * amp
    r = [0.0, 0.0, 0.0]
    r[axis] = ang
    pb[bone].rotation_euler = r
    pb[bone].keyframe_insert("rotation_euler", frame=f)


for f in range(1, N_FRAMES + 2):
    t = (f - 1) / N_FRAMES
    swing("thigh.L", f, 0.6, 0.0)
    swing("thigh.R", f, 0.6, math.pi)
    swing("shin.L", f, 0.5, -0.8)
    swing("shin.R", f, 0.5, math.pi - 0.8)
    swing("upper_arm.L", f, 0.5, math.pi)
    swing("upper_arm.R", f, 0.5, 0.0)
    swing("lower_arm.L", f, 0.35, math.pi - 0.5)
    swing("lower_arm.R", f, 0.35, -0.5)
    swing("spine", f, 0.08, 0.0, axis=2)
    swing("head", f, 0.12, 1.0, axis=2)
    pb["root"].location = (0, 0, abs(math.sin(t * math.tau * 2)) * 0.05)
    pb["root"].keyframe_insert("location", frame=f)
bpy.ops.object.mode_set(mode="OBJECT")
scene.frame_end = N_FRAMES + 1

# ---- studio: floor, key + fill light, camera
bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, 0))
bpy.context.object.data.materials.append(material("Floor", (0.35, 0.37, 0.4, 1), 0.9))
bpy.ops.object.light_add(type="AREA", location=(2.5, -3, 4))
key = bpy.context.object
key.data.energy, key.data.size = 700, 3
key.rotation_euler = (math.radians(55), 0, math.radians(35))
bpy.ops.object.light_add(type="AREA", location=(-3, -2, 2))
fill = bpy.context.object
fill.data.energy, fill.data.size = 200, 3
fill.rotation_euler = (math.radians(75), 0, math.radians(-50))
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.08, 0.1, 0.14, 1)
bpy.ops.object.camera_add(location=(0, -6.2, 1.6))
cam = bpy.context.object
cam.rotation_euler = (math.radians(88), 0, 0)
cam.data.lens = 50
scene.camera = cam

# ---- render + export (glb carries mesh + skin + bone animation)
scene.frame_set(4 if quick else 7)
rb.render_still("rig_still.png", engine=engine, samples=24, res=(540, 720))
rb.render_anim("rig_walk.mp4", engine=engine, samples=6, res=(360, 480), frames=(1, N_FRAMES))
rb.export("rig.glb", export_skins=True)
rb.export("rig.fbx")
rb.export("rig.blend")
info = {
    "bones": [b.name for b in arm.bones],
    "vertex_groups": [g.name for g in body.vertex_groups],
    "vertices": len(body.data.vertices),
    "unweighted_vertices": bad,
    "weighting": method,
    "keyframed_bones": sorted({fc.data_path.split('"')[1] for fc in arm_obj.animation_data.action.fcurves if "pose.bones" in fc.data_path}),
}
with open(rb.out("rig_bones.json"), "w") as fh:
    json.dump(info, fh, indent=2)
rb.report(**{"rig": info})
