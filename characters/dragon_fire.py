"""Rigged dragon breathing fire: neck, rig, skin weights, animation, fire, camera, mp4.

    python3 characters/dragon_ref/prep_glb.py refs3d/red_dragon.glb refs3d/red_dragon_head.npz
    rblender run characters/dragon_ref3d.py -- refs3d/red_dragon_head.npz --no-render   # writes the model + rig data
    cp out/dragon_ref3d_model.blend out/dragon_ref3d_rig.json exports/
    rblender run characters/dragon_fire.py -- [--quick] [--poses] [--still F] [--frames N] [--palette ember] [--no-render]

The model is the head built by dragon_ref3d.py (the target had a body; this model stops at the skull). A neck and a
chest are added so the head has something to move on. Bones: root, neck1..4, head, jaw, tongue1..3, eye.R, eye.L, all
with local X = world X, so a rotation about X is a pitch. Skin weights are computed here, not painted: the skull blends
head to neck at its back, the jaw and its teeth follow the jaw bone, the tongue blends along its length, the neck blends
along its centre line. Procedural materials read a stored rest position so scales do not swim while the mesh deforms.
The fire is a volume cone parented to the head bone with animatable Intensity / Front / Flow nodes.
"""
import json
import math
import os
import time

import bmesh
import bpy
import numpy as np
import prec
import rb
from mathutils import Vector as V

opts = rb.args()
QUICK = "--quick" in opts
NOREND = "--no-render" in opts
POSES = "--poses" in opts
STILL = int(opts[opts.index("--still") + 1]) if "--still" in opts else None
PAL = opts[opts.index("--palette") + 1] if "--palette" in opts else "ember"
FPS = 24
N = int(opts[opts.index("--frames") + 1]) if "--frames" in opts else 168
RES = (640, 360) if QUICK else (1280, 720)
SAMPLES = 12 if QUICK else 16
pal = prec.DRAGON_PALETTES[PAL]
T0 = time.time()
LOG = {"seconds": {}}
rep = prec.Report()


def lap(name):
    LOG["seconds"][name] = round(time.time() - T0, 1)


def find(path):
    for c in (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{path}: run characters/dragon_ref3d.py first and copy its outputs to exports/")


def frame_of(seconds):
    return 1 + int(round(seconds * FPS))


RIG = json.load(open(find("exports/dragon_ref3d_rig.json")))
bpy.context.preferences.edit.use_global_undo = False
scene = rb.clean()

# ------------------------------------------------------------------------------------------------ load the model
with bpy.data.libraries.load(find("exports/dragon_ref3d_model.blend"), link=False) as (_src, _dst):
    _dst.objects = list(_src.objects)
for _o in _dst.objects:
    scene.collection.objects.link(_o)
O = {o.name: o for o in _dst.objects}
skull, jaw, horn, tongue = O["Skull"], O["Jaw"], O["Horn"], O["Tongue"]
spikes = [O[n] for n in RIG["spikes"]]
teeth = [O[n] for n in RIG["teeth"]]
eyes = [O["EyeR"], O["EyeL"]]
lids = [O["LidR"], O["LidL"]]
print("LOADED", len(O), "objects", flush=True)

# ------------------------------------------------------------------------------------------------ neck and chest
sb = RIG["skull_back"]
by = sb["y"]
zc = (sb["zt"] + sb["zb"]) / 2
NECK_P = np.array([[0.0, by - 0.06 + i * 0.26, zc - d] for i, d in enumerate([0.0, 0.05, 0.25, 0.58, 0.95])])  # P0 at the skull, P4 below the floor
NECK_R = np.array([0.14, 0.17, 0.22, 0.29, 0.38])


def catmull(P, n):
    """Catmull-Rom curve through the control points P (K,3), n points."""
    K = len(P)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for u in np.linspace(0, K - 1, n):
        i = min(int(u), K - 2)
        t = u - i
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


dense = catmull(NECK_P, 48)
dense_r = np.interp(np.linspace(0, 4, 48), np.arange(5), NECK_R)
neck = prec.tube("Neck", [tuple(p) for p in dense], list(dense_r), [], n=40, cap_start=True, cap_end=True)

# ------------------------------------------------------------------------------------------------ materials
SKINNED = [skull, jaw, horn, tongue, neck] + spikes + teeth + eyes + lids
for o in SKINNED:
    prec.add_rest_attribute(o)
horn_r = np.linalg.norm(prec.coords(horn), axis=1)
m_skin = prec.scale_skin_material("Skin", pal, orco="rest")
m_skin_neck = prec.scale_skin_material("SkinNeck", pal, cell=0.038, bump=0.8, orco="rest")  # a body this size wants bigger scales
m_horn = prec.horn_material("Horn", pal, r0=float(horn_r.min()), r1=float(horn_r.max()), orco="rest")
m_mouth = prec.flesh_material("Mouth", pal["mouth"], orco="rest")
m_tooth = prec.simple_material("Tooth", pal["tooth"], rough=0.3, spec=0.6)
EYE_MATS = []
for _e, _d in zip(eyes, RIG["eyes"]):
    g_ = np.array(_d["gaze"])
    h_ = np.cross(g_, [0.0, 0.0, 1.0])
    h_ /= np.linalg.norm(h_)
    u_ = np.cross(h_, g_)
    EYE_MATS.append(prec.eye_material(f"Eye_{_e.name}", pal, g_, h_, u_, _d["radius"], orco="rest"))
for o, ms in ([(skull, [m_skin, m_mouth]), (jaw, [m_skin, m_mouth]), (neck, [m_skin_neck]), (tongue, [m_mouth])]
              + [(o, [m_horn]) for o in [horn] + spikes] + [(o, [m_tooth]) for o in teeth] + [(o, [m_skin]) for o in lids]
              + list(zip(eyes, [[m] for m in EYE_MATS]))):
    o.data.materials.clear()
    for m in ms:
        o.data.materials.append(m)
lap("model")

# ------------------------------------------------------------------------------------------------ armature
arm_d = bpy.data.armatures.new("DragonRig")
arm = bpy.data.objects.new("DragonRig", arm_d)
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")


def add_bone(name, head, tail, parent=None, connect=False):
    b = arm_d.edit_bones.new(name)
    b.head, b.tail = tuple(head), tuple(tail)
    d = V(tail) - V(head)
    d.normalize()
    b.align_roll((0.0, -d.z, d.y))  # local X = world X, so a rotation about X is a pitch
    if parent:
        b.parent = arm_d.edit_bones[parent]
        b.use_connect = connect
    return b


P = NECK_P
add_bone("root", P[4] + np.array([0.0, 0.30, 0.0]), P[4])
add_bone("neck1", P[4], P[3], "root", True)
add_bone("neck2", P[3], P[2], "neck1", True)
add_bone("neck3", P[2], P[1], "neck2", True)
add_bone("neck4", P[1], P[0], "neck3", True)
add_bone("head", P[0], (0.0, -0.25, 0.05), "neck4", True)
hj = RIG["jaw"]["hinge_yz"]
tj = RIG["jaw"]["tip_yz"]
add_bone("jaw", (0.0, hj[0], hj[1]), (0.0, tj[0], tj[1]), "head")
tg = RIG["tongue"]
to = np.array(tg["origin_yz"])
ta = np.array(tg["axis_yz"])
TL = tg["length"]
tp = [to + ta * TL * f for f in (1.0, 2 / 3, 1 / 3, 0.0)]  # root -> tip
add_bone("tongue1", (0.0, *tp[0]), (0.0, *tp[1]), "jaw")
add_bone("tongue2", (0.0, *tp[1]), (0.0, *tp[2]), "tongue1", True)
add_bone("tongue3", (0.0, *tp[2]), (0.0, *tp[3]), "tongue2", True)
for _nm, _d in zip(("eye.R", "eye.L"), RIG["eyes"]):
    add_bone(_nm, _d["centre"], np.array(_d["centre"]) + np.array(_d["gaze"]) * 0.03, "head")
bpy.ops.object.mode_set(mode="OBJECT")
for pb in arm.pose.bones:
    pb.rotation_mode = "XYZ"
print("BONES", [b.name for b in arm_d.bones], flush=True)
print("BONE_X", {b.name: [round(float(x), 2) for x in b.matrix_local.col[0][:3]] for b in arm_d.bones if b.name in ("head", "jaw", "neck2")}, flush=True)

# ------------------------------------------------------------------------------------------------ skin weights
ss = prec.smoothstep


def write_weights(ob, W):
    names = [n for n in W]
    Mx = np.stack([np.asarray(W[n], float) for n in names], axis=1)
    tot = Mx.sum(1, keepdims=True)
    Mx = np.where(tot > 1e-9, Mx / np.maximum(tot, 1e-9), 0.0)
    for k, n in enumerate(names):
        vg = ob.vertex_groups.new(name=n)
        for i in np.nonzero(Mx[:, k] > 1e-4)[0]:
            vg.add([int(i)], float(Mx[i, k]), "REPLACE")
    return int((Mx.sum(1) < 0.5).sum())


def seg_param(Pt, poly):
    best_s = np.zeros(len(Pt))
    best_d = np.full(len(Pt), 1e9)
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        t = np.clip(((Pt - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(Pt - (a + t[:, None] * ab), axis=1)
        m = d < best_d
        best_d[m], best_s[m] = d[m], i + t[m]
    return best_s


UNWEIGHTED = 0


def weigh(ob, W):
    global UNWEIGHTED
    UNWEIGHTED += write_weights(ob, W)


def zeros(ob):
    return np.zeros(len(ob.data.vertices))


# skull: head, blending to neck4 over its back 0.3 m
Ps = prec.coords(skull)
u_back = ss(by - 0.30, by - 0.02, Ps[:, 1])
weigh(skull, {"head": 1 - u_back, "neck4": u_back})
# jaw and the lower teeth follow the jaw bone; tongue blends along its length and joins the jaw at its root
weigh(jaw, {"jaw": np.ones(len(jaw.data.vertices))})
Pt_ = prec.coords(tongue)
tf = np.clip(((Pt_[:, 1] - to[0]) * ta[0] + (Pt_[:, 2] - to[1]) * ta[1]) / TL, 0, 1)  # 0 tip .. 1 root
hat = lambda c: np.clip(1 - np.abs(tf - c) / (1 / 3), 0, 1)
weigh(tongue, {"tongue1": hat(5 / 6), "tongue2": hat(0.5), "tongue3": hat(1 / 6), "jaw": ss(0.78, 0.98, tf)})
# rigid parts
weigh(horn, {"head": np.ones(len(horn.data.vertices))})
tree_j, tree_s = prec.bvh_of(jaw), prec.bvh_of(skull)
SPIKE_ON_JAW = 0
for sp in spikes:
    c = np.mean(prec.coords(sp), axis=0)
    dj = tree_j.find_nearest(V(c))[3]
    ds = tree_s.find_nearest(V(c))[3]
    onj = dj < ds
    SPIKE_ON_JAW += int(onj)
    weigh(sp, {"jaw" if onj else "head": np.ones(len(sp.data.vertices))})
for t_ in teeth:
    weigh(t_, {"jaw" if "jaw" in t_.name else "head": np.ones(len(t_.data.vertices))})
for l_ in lids:
    weigh(l_, {"head": np.ones(len(l_.data.vertices))})
weigh(eyes[0], {"eye.R": np.ones(len(eyes[0].data.vertices))})
weigh(eyes[1], {"eye.L": np.ones(len(eyes[1].data.vertices))})
# the neck along its centre line
for ob in (neck,):
    Pn = prec.coords(ob)
    s = seg_param(Pn, NECK_P)
    W = {"head": 1 - ss(0.0, 0.9, s), "root": ss(3.0, 4.0, s)}
    for i, bn in enumerate(("neck4", "neck3", "neck2", "neck1")):
        W[bn] = np.clip(1 - np.abs(s - (i + 0.5)), 0, 1)
    weigh(ob, W)
for ob in SKINNED:
    m = ob.modifiers.new("Armature", "ARMATURE")
    m.object = arm
    m.use_vertex_groups = True
    ob.parent = arm
lap("rig")
rep.count("rig", "vertices without a skin weight", UNWEIGHTED, 0)
rep.count("rig", "bones", len(arm_d.bones), 12)
print(f"RIG bones={len(arm_d.bones)} skinned objects={len(SKINNED)} spikes on jaw={SPIKE_ON_JAW}/{len(spikes)} unweighted={UNWEIGHTED}", flush=True)


# ------------------------------------------------------------------------------------------------ posing
def eval_coords(ob):
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    a = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", a)
    ev.to_mesh_clear()
    return a.reshape(-1, 3)


def set_pose(raise_deg=0.0, jaw_deg=0.0, tongue=0.0, back=0.0):
    sg_r, sg_j = RAISE_SIGN, JAW_SIGN
    splits = {"neck1": 0.08, "neck2": 0.17, "neck3": 0.22, "neck4": 0.23, "head": 0.30}
    for n, f in splits.items():
        arm.pose.bones[n].rotation_euler = (math.radians(sg_r * raise_deg * f), 0, 0)
    arm.pose.bones["jaw"].rotation_euler = (math.radians(sg_j * jaw_deg), 0, 0)
    for k, n in enumerate(("tongue1", "tongue2", "tongue3")):
        arm.pose.bones[n].rotation_euler = (math.radians(sg_j * tongue * (0.5 + 0.5 * k)), 0, 0)
    arm.pose.bones["root"].location = (0.0, -back, 0.0)  # local +Y is forward (-y world): negative = backwards
    bpy.context.view_layer.update()


RAISE_SIGN = JAW_SIGN = 1.0
rest_tip = eval_coords(teeth[[i for i, t in enumerate(teeth) if "jaw" in t.name][0]])[:, 2].min()
set_pose(jaw_deg=30)
JAW_SIGN = 1.0 if eval_coords(teeth[[i for i, t in enumerate(teeth) if "jaw" in t.name][0]])[:, 2].min() < rest_tip else -1.0
set_pose()
nose_z = eval_coords(skull)[:, 1].argmin()
rest_nose = eval_coords(skull)[nose_z, 2]
set_pose(raise_deg=30)
RAISE_SIGN = 1.0 if eval_coords(skull)[nose_z, 2] > rest_nose else -1.0
set_pose(raise_deg=30)
raised = eval_coords(skull)[nose_z, 2] - rest_nose
set_pose(jaw_deg=30)
opened = rest_tip - eval_coords(teeth[[i for i, t in enumerate(teeth) if "jaw" in t.name][0]])[:, 2].min()
set_pose()
rep.at_least("rig", "30 degrees of jaw lowers the lower teeth (cm)", opened * 100, 5.0)
rep.at_least("rig", "30 degrees of raise lifts the nose (cm)", raised * 100, 10.0)
rep.check("rig", "armature modifier after the mirror on every skinned mesh", all([m.type for m in o.modifiers][-1] == "ARMATURE" for o in SKINNED))
print(f"SIGNS jaw={JAW_SIGN} raise={RAISE_SIGN} opened_cm={opened * 100:.1f} raised_cm={raised * 100:.1f}", flush=True)
lap("pose_test")

# ------------------------------------------------------------------------------------------------ scene
FLOOR_Z = -0.80
prec.studio(top=(0.012, 0.014, 0.028), bottom=(0.075, 0.022, 0.008), floor=True, floor_z=FLOOR_Z, floor_r=9.0)  # a red horizon glow
_fm = bpy.data.objects["StudioFloor"].data.materials[0].node_tree.nodes["Principled BSDF"]
_fm.inputs["Base Color"].default_value = (0.025, 0.022, 0.020, 1.0)
_fm.inputs["Roughness"].default_value = 0.55  # matte rock with a hint of gloss for the fire
_rng = np.random.default_rng(11)
_rock = prec.simple_material("Rock", (0.030, 0.026, 0.024), rough=0.7, spec=0.2)
for _k in range(18):  # faceted boulders in an arc behind the pit: depth, and something for the fire light to catch
    _az = math.radians(_rng.uniform(100, 290))
    _rad = _rng.uniform(2.8, 5.2)
    _s = _rng.uniform(0.35, 1.1)
    _bm = bmesh.new()
    bmesh.ops.create_icosphere(_bm, subdivisions=2, radius=1.0)
    for _v in _bm.verts:
        _v.co = V((_v.co.x * _s * _rng.uniform(0.9, 1.5), _v.co.y * _s * _rng.uniform(0.9, 1.5), _v.co.z * _s * 0.7)) * _rng.uniform(0.82, 1.18)
    _r = prec.new_object(f"Rock{_k}", _bm, [_rock], smooth=False)
    _r.location = (_rad * math.cos(_az), _rad * math.sin(_az), FLOOR_Z + 0.18 * _s)
prec.area_light("Key", (1.7, -1.9, 1.1), (0, 0, 0.1), 60, 1.4, pal["lights"]["Key"])
prec.area_light("Rim", (-1.3, 1.7, 0.9), (0, 0, 0.1), 110, 1.0, (0.55, 0.70, 1.0))

# the fire: a volume cone from the back of the mouth, parented to the head bone so it follows every pitch
MC = np.array(RIG["mouth_c"]) + np.array([0.0, 0.05, 0.0])
set_pose()
scene.frame_set(1)
BR = dict(pal["breath"])
BR["emission"] = BR["emission"] * 1.0
BR["density"] = BR["density"] * 1.15
breath = prec.breath_volume("Breath", tuple(MC), (0.0, -1.0, -0.22), 1.9, 0.62, BR, r_start=0.10, flow_speed=0.22)
fire_light_d = bpy.data.lights.new("FireLight", "POINT")
fire_light_d.color = pal["breath"]["light"]
fire_light_d.energy = 0.0
fire_light_d.volume_factor = 0.0  # a light inside a volume scatters into a bright dot
fire_light = bpy.data.objects.new("FireLight", fire_light_d)
scene.collection.objects.link(fire_light)
fire_light.location = tuple(MC + np.array([0.0, -0.45, -0.05]))


def parent_to_head(o):
    o.parent = arm
    o.parent_type = "BONE"
    o.parent_bone = "head"
    o.matrix_parent_inverse.identity()
    bpy.context.view_layer.update()
    # Blender parents to a bone's TAIL: take the parent matrix it actually uses, so the child stays where it is at rest
    o.matrix_parent_inverse = (o.matrix_world @ o.matrix_basis.inverted()).inverted()


parent_to_head(breath)
parent_to_head(fire_light)

# embers: particles from a small plane in the flame path, rising slowly, emitting only while the jet runs
_ember_mat = prec.simple_material("Ember", (1.0, 0.30, 0.03), rough=0.5, emission=(1.0, 0.35, 0.05))
_ember_mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 12.0
bm_e = bmesh.new()
bmesh.ops.create_icosphere(bm_e, subdivisions=1, radius=0.010)
ember_mesh = prec.new_object("EmberMesh", bm_e, [_ember_mat])
ember_mesh.location = (0.0, 0.0, -20.0)  # the instanced object itself stays far away
bm_p = bmesh.new()
bmesh.ops.create_grid(bm_p, x_segments=2, y_segments=2, size=0.025)
emitter = prec.new_object("EmberEmitter", bm_p, [])
emitter.location = tuple(MC + np.array([0.0, -0.32, -0.08]))
emitter.rotation_euler = V((0.0, -1.0, -0.22)).to_track_quat("Z", "Y").to_euler()
parent_to_head(emitter)
_ps = emitter.modifiers.new("Embers", "PARTICLE_SYSTEM").particle_system.settings
_ps.count = 500 if not QUICK else 160
_ps.frame_start, _ps.frame_end = frame_of(2.95), frame_of(5.4)
_ps.lifetime, _ps.lifetime_random = 70, 0.6
_ps.emit_from = "FACE"
_ps.normal_factor = 2.4
_ps.factor_random = 1.6
_ps.render_type = "OBJECT"
_ps.instance_object = ember_mesh
_ps.particle_size, _ps.size_random = 1.0, 0.7
_ps.effector_weights.gravity = -0.12  # slightly buoyant
emitter.show_instancer_for_render = False
bpy.context.view_layer.update()

# camera on a tracking target
cam_target = bpy.data.objects.new("CamTarget", None)
scene.collection.objects.link(cam_target)
cam = prec.camera((1.9, -1.8, 0.5), (0, 0, 0), 40, "FireCam")
tr = cam.constraints.new("TRACK_TO")
tr.target = cam_target
tr.track_axis = "TRACK_NEGATIVE_Z"
tr.up_axis = "UP_Y"
scene.camera = cam

# ------------------------------------------------------------------------------------------------ animation
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 1, N
POSE_KEYS = [  # seconds, head raise (deg), jaw (deg), tongue curl, body recoil (m)
    (0.0, 0, 3, 0, 0.0), (1.0, 3, 3, 0, 0.0), (1.8, 30, 10, 4, 0.04), (2.5, 42, 16, 8, 0.07), (2.9, -14, 40, -6, -0.04),
    (3.6, -9, 37, 0, -0.02), (4.5, -16, 40, 0, -0.03), (5.4, -6, 34, 0, 0.0), (6.0, 4, 16, 0, 0.0), (7.0, 2, 4, 0, 0.0)]
for t_, r_, j_, tg_, b_ in POSE_KEYS:
    set_pose(raise_deg=r_, jaw_deg=j_, tongue=tg_, back=b_)
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_euler", frame=frame_of(t_))
    arm.pose.bones["root"].keyframe_insert("location", frame=frame_of(t_))
set_pose()
act = arm.animation_data.action


def noise(fc, strength, scale, f0, f1, ramp=8):
    m = fc.modifiers.new("NOISE")
    m.strength, m.scale = strength, scale
    m.use_restricted_range = True
    m.frame_start, m.frame_end = f0, f1
    m.blend_in = m.blend_out = ramp


for fc in act.fcurves:  # the body shakes while the fire roars
    if fc.data_path.endswith("rotation_euler") and fc.array_index == 0:
        who = fc.data_path.split('"')[1]
        if who in ("neck2", "neck3", "neck4", "head"):
            noise(fc, 0.018 if who == "head" else 0.010, 4.0, frame_of(3.0), frame_of(5.5))
        if who in ("tongue2", "tongue3"):
            noise(fc, 0.20, 5.0, frame_of(2.9), frame_of(5.4))
        if who == "jaw":
            noise(fc, 0.025, 3.0, frame_of(3.0), frame_of(5.4))

# fire controls: a faint glow in the mouth while the head rears back, then the jet
mat_b = breath.data.materials[0]
ntb = mat_b.node_tree


def key_socket(node, t_, v_):
    node.outputs[0].default_value = v_
    node.outputs[0].keyframe_insert("default_value", frame=frame_of(t_))


for t_, v_ in ((0.0, 0.0), (2.1, 0.0), (2.5, 0.25), (2.85, 0.45), (3.05, 1.0), (5.3, 1.0), (5.9, 0.0), (7.0, 0.0)):
    key_socket(ntb.nodes["Intensity"], t_, v_)
for t_, v_ in ((0.0, 0.0), (2.1, 0.0), (2.5, 0.15), (2.9, 0.18), (3.8, 1.9), (7.0, 1.9)):
    key_socket(ntb.nodes["Front"], t_, v_)
for fc in ntb.animation_data.action.fcurves:
    if "Intensity" in fc.data_path:
        noise(fc, 0.30, 3.0, frame_of(3.0), frame_of(5.4))
for t_, v_ in ((0.0, 0.0), (2.1, 0.0), (2.5, 25.0), (3.05, 380.0), (5.3, 380.0), (5.9, 0.0), (7.0, 0.0)):
    fire_light_d.energy = v_
    fire_light_d.keyframe_insert("energy", frame=frame_of(t_))
for fc in fire_light_d.animation_data.action.fcurves:
    noise(fc, 90.0, 3.0, frame_of(3.0), frame_of(5.4))
for em in EYE_MATS:  # the eyes flare while the dragon rears and roars
    for t_, v_ in ((0.0, 0.5), (1.8, 1.2), (2.5, 3.2), (3.0, 2.4), (5.4, 1.6), (6.4, 0.6)):
        key_socket(em.node_tree.nodes["EyeGlow"], t_, v_)
for t_, loc_, tgt_, lens_ in (
        (0.0, (1.6, -1.5, 0.45), (0.0, 0.15, 0.05), 46), (1.8, (1.5, -1.5, 0.45), (0.0, 0.10, 0.25), 40),
        (2.5, (1.6, -1.7, 0.35), (0.0, 0.05, 0.42), 32), (3.0, (1.3, -1.5, -0.10), (0.0, -0.45, 0.00), 26),
        (4.5, (2.3, -1.3, 0.25), (0.0, -0.75, 0.00), 28), (5.5, (2.6, -0.6, 0.30), (0.0, -0.60, 0.00), 30),
        (7.0, (2.4, 0.4, 0.35), (0.0, -0.20, 0.05), 36)):
    cam.location = loc_
    cam.keyframe_insert("location", frame=frame_of(t_))
    cam_target.location = tgt_
    cam_target.keyframe_insert("location", frame=frame_of(t_))
    cam.data.lens = lens_
    cam.data.keyframe_insert("lens", frame=frame_of(t_))
for fc in cam.animation_data.action.fcurves:  # handheld shake while the fire roars
    noise(fc, 0.012, 2.5, frame_of(2.9), frame_of(5.6), ramp=6)
lap("animation")
rep.check("animation", "pose keys on every bone", len({fc.data_path.split('"')[1] for fc in act.fcurves if "pose.bones" in fc.data_path}) >= 10,
          f"{len(act.fcurves)} curves")
rep.check("animation", "fire intensity, front, light and eye glow are keyed",
          all(x.animation_data and x.animation_data.action for x in (ntb, fire_light_d, EYE_MATS[0].node_tree)))

# ------------------------------------------------------------------------------------------------ render setup
eng = "BLENDER_EEVEE_NEXT"
scene.render.engine = eng
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.eevee.taa_render_samples = SAMPLES
for _k, _v in (("use_volumetric_shadows", True), ("volumetric_tile_size", "4" if QUICK else "2"), ("volumetric_samples", 48 if not QUICK else 24),
               ("use_shadows", True)):
    try:
        setattr(scene.eevee, _k, _v)
    except (AttributeError, TypeError) as _e:
        print("EEVEE setting skipped:", _k, _e, flush=True)
scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items] else "Standard"
prec.set_grade(scene, **pal["grade"], glare=dict(threshold=0.9, size=8, mix=-0.35))
scene.render.image_settings.file_format = "PNG"


def still(name, frame):
    scene.frame_set(frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = rb.out(name)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"FRAME {frame} {time.time() - t:.1f}s", flush=True)
    return scene.render.filepath


def synth_audio(path, seconds, sr=44100):
    """Rumble, an inhale, a roar and a fire roar that follows the intensity curve: filtered noise and a sawtooth with
    vibrato. Stereo WAV, 16 bit."""
    import wave
    rng = np.random.default_rng(7)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = np.fft.rfftfreq(n, 1 / sr)

    def band(x, lo, hi):
        X = np.fft.rfft(x)
        X *= ((f >= lo) & (f <= hi)) * 1.0 / (1 + (f / max(hi, 1)) ** 4)
        return np.fft.irfft(X, n)

    def env(points):
        ts, a = zip(*points)
        return np.interp(t, ts, a)
    noise = lambda: rng.standard_normal(n)
    rumble = band(noise(), 20, 110) * 6.0 * env([(0, 0.4), (2.5, 0.9), (3.0, 1.2), (5.4, 1.0), (7.0, 0.4)])
    whoosh = band(noise(), 250, 2600) * 1.4 * env([(1.2, 0), (2.4, 0.9), (2.9, 0.1), (3.0, 0)])
    f0 = 95 - 28 * np.clip((t - 2.9) / 2.5, 0, 1)  # the roar's pitch falls
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.025 * np.sin(2 * np.pi * 6.5 * t))) / sr
    saw = 2 * ((phase / (2 * np.pi)) % 1) - 1
    roar = band(saw + 0.8 * noise(), 70, 1800) * 2.2 * env([(2.85, 0), (2.95, 1.0), (3.6, 0.8), (5.0, 0.35), (5.6, 0)])
    fire_amp = env([(2.9, 0), (3.1, 0.9), (5.3, 0.9), (5.9, 0)])
    fire = band(noise(), 120, 1700) * 2.0 * fire_amp
    clicks = (rng.random(n) < 0.0014 * fire_amp).astype(float) * rng.standard_normal(n)
    crackle = band(clicks * 40, 1500, 9000) * fire_amp
    thump = np.sin(2 * np.pi * 48 * t) * np.exp(-np.clip(t - 2.9, 0, None) * 7) * (t >= 2.9) * 1.6
    mono = rumble + whoosh + roar + fire + crackle + thump
    mono /= np.abs(mono).max() / 0.85
    left = mono * (0.9 + 0.1 * np.sin(2 * np.pi * 0.2 * t))
    right = mono * (0.9 - 0.1 * np.sin(2 * np.pi * 0.2 * t))
    pcm = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


if "--export" in opts:
    # the whole scene as a .blend (procedural materials, fire, camera), then a glb of the rig and the skinned meshes with
    # flat PBR colours (procedural shaders and volumes are not glTF); animation is baked per frame so the shake is kept
    scene.frame_set(1)
    rb.export("dragon_fire_rigged.blend")
    flat = {"skin": prec.simple_material("G_Skin", pal["side"], rough=0.45, spec=0.5),
            "mouth": prec.simple_material("G_Mouth", pal["mouth"], rough=0.3, spec=0.6),
            "horn": prec.simple_material("G_Horn", pal["horn_mid"], rough=0.45, spec=0.4),
            "tooth": prec.simple_material("G_Tooth", pal["tooth"], rough=0.3, spec=0.6)}
    for o, ms in ([(skull, [flat["skin"], flat["mouth"]]), (jaw, [flat["skin"], flat["mouth"]]), (neck, [flat["skin"]]), (tongue, [flat["mouth"]])]
                  + [(o, [flat["horn"]]) for o in [horn] + spikes] + [(o, [flat["tooth"]]) for o in teeth] + [(o, [flat["skin"]]) for o in lids]
                  + [(o, [prec.simple_material(f"G_Eye{i}", pal["eye"], rough=0.15, spec=0.8, emission=pal["eye"])]) for i, o in enumerate(eyes)]):
        o.data.materials.clear()
        for m in ms:
            o.data.materials.append(m)
    for o in bpy.data.objects:
        o.select_set(False)
    for o in [arm] + SKINNED:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        gl = rb.export("dragon_fire_rigged.glb", use_selection=True, export_apply=True, export_bake_animation=True)
    except TypeError:
        gl = rb.export("dragon_fire_rigged.glb", use_selection=True, export_apply=True)
    rep.check("export", "rigged glb written", os.path.getsize(gl) > 100000, f"{os.path.getsize(gl)} bytes")
    lap("export")
if POSES or STILL is not None:
    frames = [STILL] if STILL is not None else [frame_of(0.5), frame_of(2.5), frame_of(3.5), frame_of(5.8)]
    outs = [still(f"_pose_{f:03d}.png", f) for f in frames]
    if len(outs) > 1:
        prec.contact_sheet(outs, rb.out("fire_poses.png"))
    lap("stills")
elif not NOREND:
    scene.frame_step = 3 if QUICK else 1
    wav = synth_audio(rb.out("dragon_fire_audio.wav"), N / FPS)
    if QUICK or "--mp4" in opts:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.filepath = rb.out("dragon_fire.mp4")
        se = scene.sequence_editor_create()
        strips = getattr(se, "strips", None) or se.sequences
        strips.new_sound("DragonAudio", wav, 1, 1)
        scene.render.ffmpeg.audio_codec = "AAC"
        scene.render.ffmpeg.audio_bitrate = 192
        scene.render.ffmpeg.audio_channels = "STEREO"
        scene.render.ffmpeg.audio_mixrate = 44100
    else:
        # a numbered JPEG sequence: a failure late in a long render still leaves every finished frame, and the mp4 is
        # assembled afterwards:  ffmpeg -framerate 24 -i out/frames/f_%04d.jpg -i out/dragon_fire_audio.wav ...
        os.makedirs(os.path.dirname(rb.out("frames/x")), exist_ok=True)
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = 95
        scene.render.filepath = rb.out("frames/f_")
    bpy.ops.render.render(animation=True)
    lap("video")

print(rep.table())
with open(rb.out("fire_report.json"), "w") as fh:
    json.dump({"rows": rep.rows, "ok": rep.ok(), "log": LOG, "frames": N, "fps": FPS, "res": RES, "palette": PAL}, fh, indent=1, default=float)
with open(rb.out("fire_report.txt"), "w") as fh:
    fh.write(rep.table() + "\n")
rb.report(fire={"checks": len(rep.rows), "pass": int(sum(r["ok"] for r in rep.rows)), "ok": bool(rep.ok()), "seconds": LOG["seconds"]})
