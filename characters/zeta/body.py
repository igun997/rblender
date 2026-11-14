"""Skin: torso, legs, arms, hands, head, ears, and the face (eyes, lashes, brows, mouth) as
surface-projected decals so blinking / talking can be done with bone translations."""
import math

import bmesh
import numpy as np
from mathutils import Matrix, Vector as V

from . import layout as L
from . import skeleton as SK
from .common import (OBJECTS, Surface, catmull, disc_decal, join, loft, loft_bm, new_object, ribbon_decal,
                     rigid_weights, skin_weights, smoothstep)

HEAD_SURFACE = None


def _profile(table, t):
    """table: [(t, rx, ry)] -> interpolated rx, ry."""
    ts = [r[0] for r in table]
    return (float(np.interp(t, ts, [r[1] for r in table])), float(np.interp(t, ts, [r[2] for r in table])))


def torso(M):
    z = [r[0] for r in L.TORSO]
    centers = [(0, r[3], r[0]) for r in L.TORSO]
    ob = loft("Skin_Torso", centers, [r[1] for r in L.TORSO], [r[2] for r in L.TORSO], mat=M["skin"], n=40)
    skin_weights(ob, SK.segments(["hips", "spine", "chest", "neck", "clavicle.L", "clavicle.R", "thigh.L", "thigh.R"]),
                 ["hips", "spine", "chest", "neck", "clavicle.L", "clavicle.R", "thigh.L", "thigh.R"], k=3.0)
    return ob


LEG_PROFILE = [(0, .078, .082), (.10, .076, .080), (.30, .066, .068), (.46, .050, .052), (.52, .047, .048),
               (.60, .050, .053), (.66, .052, .055), (.82, .036, .038), (1.0, .030, .032)]
ARM_PROFILE = [(0, .041, .043), (.12, .038, .040), (.45, .033, .035), (.52, .029, .030), (.62, .031, .033),
               (.85, .025, .022), (1.0, .021, .017)]


def _limb(name, pts, profile, n, mat, count=22, side=(1, 0, 0), extend_start=0.0):
    path = catmull(pts, 14)
    # resample evenly
    from .common import lerp_path
    path = lerp_path(path, count)
    rx, ry = zip(*[_profile(profile, i / (count - 1)) for i in range(count)])
    return loft(name, path, list(rx), list(ry), mat=mat, n=n, side=side)


def legs(M):
    for sd, s in L.SIDES.items():
        p = L.leg_pts(s)
        ob = _limb(f"Skin_Leg.{sd}", [p["hip"] + V((0, 0, 0.02)), p["knee"], p["ankle"] - V((0, 0, 0.03))], LEG_PROFILE, 28, M["skin"])
        names = ["hips", f"thigh.{sd}", f"shin.{sd}", f"foot.{sd}"]
        skin_weights(ob, SK.segments(names), names, k=4.0)


def arms(M):
    for sd, s in L.SIDES.items():
        a = L.arm_pts(s)
        start = a["clav"].lerp(a["shoulder"], 0.55)
        ob = _limb(f"Skin_Arm.{sd}", [start, a["shoulder"].lerp(a["elbow"], 0.25), a["elbow"], a["wrist"]], ARM_PROFILE, 20, M["skin"], side=(0, 1, 0))
        names = ["chest", f"clavicle.{sd}", f"upper_arm.{sd}", f"forearm.{sd}", f"hand.{sd}"]
        skin_weights(ob, SK.segments(names), names, k=4.0)


def hands(M):
    for sd, s in L.SIDES.items():
        g = L.hand_geometry(s)
        parts = []
        # palm
        palm = loft(f"Skin_Palm.{sd}", g["palm"], [0.026, 0.031, 0.033, 0.030], [0.013, 0.012, 0.0115, 0.010],
                    mat=M["skin"], n=14, side=(0, 1, 0))
        parts.append(palm)
        fb = {}
        for fn, pts in g["fingers"].items():
            r0, r1 = (0.0125, 0.0085) if fn == "thumb" else (0.0105, 0.0072)
            if fn == "pinky":
                r0, r1 = 0.0092, 0.0064
            f = loft(f"Skin_{fn}.{sd}", pts, [r0, r0 * 0.95, r1 * 1.05, r1 * 0.8], [r0, r0 * 0.95, r1 * 1.05, r1 * 0.8],
                     mat=M["skin"], n=10, side=(0, 1, 0))
            parts.append(f)
        hand = join(f"Skin_Hand.{sd}", parts)
        names = [f"forearm.{sd}", f"hand.{sd}"] + SK.finger_bones(sd)
        skin_weights(hand, SK.segments(names), names, k=4.0)


def head(M):
    global HEAD_SURFACE
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=56, v_segments=36, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co
        zt = max(0.0, -z)
        xs = 1 - 0.30 * zt ** 1.6
        ys = 1 - 0.16 * zt ** 1.2
        if y > 0:
            ys *= 1.05
        # soften the chin
        zz = z * (1 - 0.03 * zt)
        v.co = V((x * xs * L.HEAD_R[0], y * ys * L.HEAD_R[1], zz * L.HEAD_R[2])) + L.HEAD_C
    ob = new_object("Skin_Head", bm, M["skin"])
    rigid_weights(ob, "head")
    HEAD_SURFACE = Surface(ob)
    # neck-to-head blending is handled by the torso loft reaching into the skull
    # ears
    for sd, s in L.SIDES.items():
        b = bmesh.new()
        bmesh.ops.create_uvsphere(b, u_segments=16, v_segments=10, radius=1.0)
        for v in b.verts:
            v.co = V((v.co.x * 0.010, v.co.y * 0.016, v.co.z * 0.028)) + V((s * 0.086, 0.008, 1.442))
        e = new_object(f"Skin_Ear.{sd}", b, M["skin"])
        rigid_weights(e, "head")
    return ob


def face(M):
    sf = HEAD_SURFACE
    cz = 1.435
    EW, EH = 0.050, 0.060
    for sd, s in L.SIDES.items():
        cx = s * 0.037
        tilt = s * 0.06
        # bones that swap open/closed eyes by sliding the decals through the skin
        add_pos = V((cx, -0.088, cz))
        SK.add_bone(f"eyeO.{sd}", add_pos, add_pos + V((0, 0.01, 0)), "head")
        SK.add_bone(f"eyeC.{sd}", add_pos, add_pos + V((0, 0.01, 0)), "head")
        center = V((cx, -0.5, cz))
        eye = disc_decal(f"Face_Eye.{sd}", None, center, EW, EH, M["eye"], rot=tilt, surface=sf, offset=0.0012, segs=36)
        rigid_weights(eye, f"eyeO.{sd}")
        a, b = EW / 2, EH / 2
        # upper lash (thick, flicked outward)
        up = []
        for k in range(11):
            phi = math.pi + 0.15 - (math.pi + 0.35) * k / 10
            up.append(V((cx + s * a * 1.04 * math.cos(phi), -0.5, cz + b * 1.02 * math.sin(phi))))
        up.append(up[-1] + V((s * 0.007, 0, -0.003)))
        widths = [0.0015, 0.0035, 0.0055, 0.0065, 0.0070, 0.0072, 0.0070, 0.0065, 0.0060, 0.0052, 0.0040, 0.0015]
        lash = ribbon_decal(f"Face_UpperLash.{sd}", None, up, widths, M["lash"], surface=sf, offset=0.0022, steps=30)
        rigid_weights(lash, f"eyeO.{sd}")
        lo = []
        for k in range(8):
            phi = math.pi + 0.35 + (math.pi - 0.7) * k / 7
            lo.append(V((cx + s * a * 0.98 * math.cos(phi), -0.5, cz + b * 0.98 * math.sin(phi))))
        low = ribbon_decal(f"Face_LowerLash.{sd}", None, lo, [0.0012, 0.0018, 0.0018, 0.0012], M["lash"], surface=sf, offset=0.0018, steps=16)
        rigid_weights(low, f"eyeO.{sd}")
        # closed eye: downward arc, parked 7 mm inside the skin until the blink slides it out
        cl = []
        for k in range(9):
            phi = math.pi + 0.2 + (math.pi - 0.4) * k / 8
            cl.append(V((cx + s * a * 1.0 * math.cos(phi), -0.5, cz - b * 0.15 + b * 0.55 * math.sin(phi) * -1.0 * -1.0)))
        # arc that bulges downward (smiling closed eye)
        cl = [V((cx + s * a * math.cos(math.pi - math.pi * k / 8), -0.5, cz + 0.004 - 0.010 * math.sin(math.pi * k / 8))) for k in range(9)]
        closed = ribbon_decal(f"Face_ClosedEye.{sd}", None, cl, [0.002, 0.004, 0.0055, 0.0065, 0.0065, 0.006, 0.005, 0.0035, 0.0015], M["lash"], surface=sf, offset=-0.0075, steps=24)
        rigid_weights(closed, f"eyeC.{sd}")
        # brow
        br = [V((cx - s * 0.020, -0.5, cz + 0.044)), V((cx + s * 0.002, -0.5, cz + 0.050)), V((cx + s * 0.024, -0.5, cz + 0.046))]
        brow = ribbon_decal(f"Face_Brow.{sd}", None, br, [0.0015, 0.0030, 0.0014], M["brow"], surface=sf, offset=0.0016, steps=14)
        rigid_weights(brow, "head")
    # nose
    nose = ribbon_decal("Face_Nose", None, [V((-0.003, -0.5, 1.402)), V((0.003, -0.5, 1.399))], [0.0022, 0.0022], M["nose"], surface=sf, offset=0.0016, steps=6)
    rigid_weights(nose, "head")
    # mouth: smile visible at rest, open mouth parked inside the skin
    mz = 1.376
    SK.add_bone("mouthS", V((0, -0.085, mz)), V((0, -0.075, mz)), "head")
    SK.add_bone("mouthO", V((0, -0.085, mz)), V((0, -0.075, mz)), "head")
    sm = [V((-0.011, -0.5, mz + 0.002)), V((-0.005, -0.5, mz - 0.001)), V((0.005, -0.5, mz - 0.001)), V((0.011, -0.5, mz + 0.002))]
    smile = ribbon_decal("Face_Smile", None, sm, [0.0012, 0.0022, 0.0022, 0.0012], M["mouth"], surface=sf, offset=0.0016, steps=14)
    rigid_weights(smile, "mouthS")
    om = disc_decal("Face_MouthOpen", None, V((0, -0.5, mz - 0.002)), 0.020, 0.016, M["mouth_in"], surface=sf, offset=-0.0075, segs=20, rings=4)
    rigid_weights(om, "mouthO")
    # join static bits is avoided on purpose: each decal keeps its own bone group


def build(M):
    torso(M)
    legs(M)
    arms(M)
    hands(M)
    head(M)
    face(M)
