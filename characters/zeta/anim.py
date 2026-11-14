"""Animation: a 5 s "Mission start!" clip, solved procedurally.

* Python forward kinematics that mirrors Blender's bone math (verified against Blender per run),
* two-bone IK for arms (salute lands the hand on the brow, follows the nodding head) and legs (feet stay planted),
* blink / mouth swaps by sliding decals through the skin,
* spring-damper secondary motion (hair, ribbon, tufts, skirt, cape) driven by head/hip acceleration.
Keys are written directly to F-curves (fast) on the armature's action.
"""
import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector as V

from . import hair as HAIR
from . import outfit as OUTFIT
from . import skeleton as SK

FPS = 24
DURATION = 5.0


def int_frames():
    return int(DURATION * FPS) + 1


def sstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def pulse(t, a, b, c, d):
    return sstep(a, b, t) * (1 - sstep(c, d, t))


def R(x=0.0, y=0.0, z=0.0):
    """Relative rotation (degrees) about armature axes X, Y, Z."""
    return Euler((math.radians(x), math.radians(y), math.radians(z)), "XYZ").to_matrix()


class Pose:
    def __init__(self, rig):
        self.rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
        self.names = [b.name for b in rig.data.bones]
        self.reset()

    def reset(self):
        self.q, self.loc, self.memo = {}, {}, {}

    # -- FK
    def pose(self, b):
        m = self.memo.get(b)
        if m is not None:
            return m
        B = Matrix.Translation(self.loc.get(b, V())) @ self.q.get(b, Quaternion()).to_matrix().to_4x4()
        p = self.parent[b]
        M = self.rest[b] @ B if p is None else self.pose(p) @ (self.rest[p].inverted() @ self.rest[b]) @ B
        self.memo[b] = M
        return M

    def r0(self, b):
        """Pose rotation of b if its own basis were identity."""
        p = self.parent[b]
        if p is None:
            return self.rest[b].to_3x3()
        return (self.pose(p) @ (self.rest[p].inverted() @ self.rest[b])).to_3x3()

    # -- setters
    def rel(self, b, Rrel):
        """Rotate b by Rrel expressed in the axes carried by its parent (rest-aligned)."""
        C = self.rest[b].to_3x3()
        self.q[b] = (C.inverted() @ Rrel @ C).to_quaternion()
        self.memo.clear()

    def move(self, b, d):
        """Translate b by world vector d (armature space)."""
        self.loc[b] = self.r0(b).inverted() @ V(d)
        self.memo.clear()

    def world_rot(self, b, Rw):
        """Make b's pose rotation equal Rw (3x3, armature space)."""
        self.q[b] = (self.r0(b).inverted() @ Rw).to_quaternion()
        self.memo.clear()

    # -- IK
    def ik2(self, upper, lower, T, pole, lens):
        l1, l2 = lens
        self.q.pop(upper, None)
        self.q.pop(lower, None)
        self.memo.clear()
        S = self.pose(upper).translation
        d = V(T) - S
        dist = min(max(d.length, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
        dirn = d.normalized()
        a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
        h = math.sqrt(max(l1 * l1 - a * a, 0.0))
        p = V(pole) - dirn * V(pole).dot(dirn)
        p = p.normalized() if p.length > 1e-6 else dirn.orthogonal().normalized()
        E = S + dirn * a + p * h
        wrist = S + dirn * dist
        for bone, tgt in ((upper, E - S), (lower, wrist - E)):
            R0 = self.r0(bone)
            d0 = R0 @ V((0, 1, 0))
            Rq = d0.rotation_difference(tgt.normalized()).to_matrix()
            self.q[bone] = (R0.inverted() @ Rq @ R0).to_quaternion()
            self.memo.clear()
        return E, wrist


def seg_len(name):
    h, t = SK.seg(name)
    return (t - h).length


def blink_amt(t):
    c = 0.0
    for t0 in (1.95, 3.35, 4.65):
        c = max(c, pulse(t, t0 - 0.06, t0 - 0.02, t0 + 0.04, t0 + 0.10))
    return c


def main_pose(P, t, rig_rest_ankle):
    """Everything except secondary motion. Returns dict of scalars used by the driver."""
    P.reset()
    breath = math.sin(math.tau * 0.28 * t)
    sal = pulse(t, 0.80, 1.55, 2.95, 3.65)
    talk = pulse(t, 1.35, 1.50, 2.75, 2.95)
    pon = pulse(t, 3.70, 3.98, 4.12, 4.62)
    sway = math.sin(math.tau * 0.21 * t)

    # hips: weight shift, settle during the salute, dip on the "pon"
    P.rel("hips", R(0.5 * breath, 1.6 * sway - 4.0 * pon, 2.0 * math.sin(math.tau * 0.17 * t) + 3.0 * sal))
    P.move("hips", V((0.008 * sway - 0.004 * sal, 0.0, -0.012 * sal - 0.045 * pon * (0.6 + 0.4 * math.sin(math.tau * 3 * t)))))
    P.rel("spine", R(1.2 * breath + 4.0 * sal + 6.0 * pon, 0.8 * sway, -4.0 * sal))
    P.rel("chest", R(-1.6 * breath + 1.5 * sal, 0.0, -2.0 * sal))
    P.rel("neck", R(1.0 * breath - 2.0 * sal + 3.0 * pon, 4.0 * sal - 5.0 * pon, 1.5 * math.sin(math.tau * 0.15 * t)))
    P.rel("head", R(1.0 * math.sin(math.tau * 0.3 * t) + 2.5 * talk * math.sin(math.tau * 3.2 * t) - 3.0 * sal + 5.0 * pon,
                    3.0 * sal - 6.0 * pon, 2.0 * math.sin(math.tau * 0.13 * t) - 2.0 * sal))
    P.rel("clavicle.L", R(0, -9.0 * sal - 0.6 * breath, 0))
    P.rel("clavicle.R", R(0, 0.6 * breath + 2.0 * pon, 0))

    # arms via IK
    head_rest_inv = P.rest["head"].inverted()
    for sd, s in (("L", 1.0), ("R", -1.0)):
        clav, ua, fa, hd = f"clavicle.{sd}", f"upper_arm.{sd}", f"forearm.{sd}", f"hand.{sd}"
        wrist_rest = SK.BONES[fa]["tail"]
        W0 = P.pose(clav) @ (P.rest[clav].inverted() @ wrist_rest)
        idle = V((0.008 * math.sin(math.tau * 0.3 * t + s), 0.006 * breath, 0.004 * breath))
        if sd == "L":
            Wsal = P.pose("head") @ (head_rest_inv @ V((0.19, -0.055, 1.45)))
            T = W0.lerp(Wsal, sal) + idle * (1 - sal)
            pole = V((s * 0.4, 1.0, -0.1)).lerp(V((1.0, 0.25, 0.35)), sal)
        else:
            lift = V((0.0, -0.03, 0.07)) * pon
            T = W0 + idle + lift
            pole = V((s * 0.4, 1.0, -0.1))
        P.ik2(ua, fa, T, pole, (seg_len(ua), seg_len(fa)))
        # wrist: flatten the hand against the brow while saluting
        # fingers: straight for the salute, relaxed otherwise; right hand grips the tablet
        for fn in SK.FINGERS:
            for k in (1, 2, 3):
                b = f"{fn}{k}.{sd}"
                curl = (-13.0 * sal + 1.5 * breath) if sd == "L" else (22.0 + 1.0 * breath)
                if fn == "thumb":
                    curl *= 0.5
                P.rel(b, R(0, s * curl, 0))

    # legs: IK keeps the feet planted on the ground
    for sd, s in (("L", 1.0), ("R", -1.0)):
        th, sh, ft, toe = f"thigh.{sd}", f"shin.{sd}", f"foot.{sd}", f"toe.{sd}"
        ankle = rig_rest_ankle[sd]
        P.ik2(th, sh, ankle, V((0.02 * s, -1.0, 0.1)), (seg_len(th), seg_len(sh)))
        P.world_rot(ft, P.rest[ft].to_3x3())
        P.world_rot(toe, P.rest[toe].to_3x3())

    # face: blink + mouth swap by sliding decals through the skin
    c = blink_amt(t)
    for sd in "LR":
        P.move(f"eyeO.{sd}", V((0, 0.0085 * c, 0)))
        P.move(f"eyeC.{sd}", V((0, -0.0095 * c, 0)))
    flap = talk * (0.5 + 0.5 * math.sin(math.tau * 3.6 * t)) + 0.9 * pon * (0.5 + 0.5 * math.sin(math.tau * 2.0 * t))
    o = 1.0 if flap > 0.5 else 0.0
    P.move("mouthS", V((0, 0.0085 * o, 0)))
    P.move("mouthO", V((0, -0.0095 * o, 0)))
    return {"sal": sal, "talk": talk, "pon": pon}


# --------------------------------------------------------------------------- secondary motion
def spring_series(drive, f0, zeta, gain, n_bones, decay=0.8, sub=4, clamp=0.55):
    """drive: (frames, 2) acceleration (x, y). Returns angles (n_bones, frames, 2) [about X, about Y]."""
    frames = len(drive)
    dt = 1.0 / FPS / sub
    out = np.zeros((n_bones, frames, 2))
    for i in range(n_bones):
        f = f0 * (decay ** i)
        w = math.tau * f
        th = np.zeros(2)
        vel = np.zeros(2)
        for k in range(frames):
            a = drive[k]
            for _ in range(sub):
                force = np.array([-gain * a[1], gain * a[0]]) * (1.0 + 0.35 * i)
                acc = -w * w * th - 2 * zeta * w * vel + force
                vel += acc * dt
                th += vel * dt
                th = np.clip(th, -clamp, clamp)
            out[i, k] = th
    return out


def build_secondary_drive(rig_rest_ankle, P, n):
    """Accelerations (armature axes) of the head-top and hips over the clip."""
    head_pos, hip_pos = [], []
    for k in range(n):
        t = k / FPS
        main_pose(P, t, rig_rest_ankle)
        tail = P.pose("head") @ (P.rest["head"].inverted() @ SK.BONES["head"]["tail"])
        head_pos.append(np.array(tail))
        hip_pos.append(np.array(P.pose("hips").translation))
    head_pos, hip_pos = np.array(head_pos), np.array(hip_pos)

    def accel(p):
        v = np.gradient(p, 1.0 / FPS, axis=0)
        a = np.gradient(v, 1.0 / FPS, axis=0)
        k = np.ones(5) / 5
        return np.stack([np.convolve(a[:, i], k, mode="same") for i in range(3)], axis=1)

    return accel(head_pos), accel(hip_pos)


def apply_secondary(P, t_idx, series):
    """series: dict name -> list of per-bone angle arrays; applies rel rotations (radians)."""
    for names, ang, mult in series:
        for i, b in enumerate(names):
            ax, ay = ang[i, t_idx] * mult
            P.rel(b, R(math.degrees(ax), math.degrees(ay), 0.0))


def build_series(head_acc, hip_acc, n):
    """All spring-driven chains: returns list of (bone_names, angles, multiplier)."""
    series = []
    drive_h = head_acc[:, :2] * 1.0 + hip_acc[:, :2] * 0.4
    # back hair: whip-like, free swinging; per-chain phase shifted wind
    for ci, chain in enumerate(HAIR.CHAINS.get("back", [])):
        wind = 0.012 * np.sin(math.tau * (0.45 * np.arange(n) / FPS) + ci * 0.7)
        d = drive_h * (0.9 + 0.1 * math.sin(ci))
        ang = spring_series(d, 2.3, 0.28, 22.0, len(chain), decay=0.86)
        ang[:, :, 1] += wind[None, :]
        series.append((chain, ang, 1.0))
    for ci, chain in enumerate(HAIR.CHAINS.get("side", [])):
        ang = spring_series(drive_h, 2.1, 0.30, 18.0, len(chain), decay=0.88)
        series.append((chain, ang, 1.0))
    for chain in HAIR.CHAINS.get("tufts", []):
        ang = spring_series(drive_h, 3.4, 0.35, 12.0, len(chain), decay=0.9, clamp=0.3)
        series.append((chain, ang, 1.0))
    for chain in HAIR.CHAINS.get("ribbon", []):
        ang = spring_series(drive_h, 3.0, 0.3, 26.0, len(chain), decay=0.8)
        series.append((chain, ang, 1.0))
    # skirt: hem flares/sways with hip motion; mapped around each chain's azimuth
    for k, chain in enumerate(OUTFIT.CHAINS.get("skirt", [])):
        th = math.tau * k / 8
        ang = spring_series(hip_acc[:, :2] * 1.0, 3.0, 0.45, 9.0, len(chain), decay=0.9, clamp=0.25)
        series.append((chain, ang, 1.0))
    for chain in OUTFIT.CHAINS.get("cape", []):
        ang = spring_series(hip_acc[:, :2] + head_acc[:, :2] * 0.2, 1.8, 0.35, 16.0, len(chain), decay=0.9)
        series.append((chain, ang, 1.0))
    return series


# --------------------------------------------------------------------------- bake
def bake(rig, name="ZetaMissionStart", duration=DURATION):
    n = int(duration * FPS) + 1
    P = Pose(rig)
    rest_ankle = {sd: SK.BONES[f"shin.{sd}"]["tail"].copy() for sd in "LR"}
    head_acc, hip_acc = build_secondary_drive(rest_ankle, P, n)
    series = build_series(head_acc, hip_acc, n)
    chans = {}
    for k in range(n):
        t = k / FPS
        main_pose(P, t, rest_ankle)
        apply_secondary(P, k, series)
        for b in P.names:
            q = P.q.get(b)
            loc = P.loc.get(b)
            if q is None and loc is None:
                continue
            ent = chans.setdefault(b, {"q": [], "loc": [], "frames": []})
            ent["frames"].append(k)
            ent["q"].append(tuple(q) if q is not None else (1.0, 0.0, 0.0, 0.0))
            ent["loc"].append(tuple(loc) if loc is not None else (0.0, 0.0, 0.0))
    # write F-curves
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    act = bpy.data.actions.new(name)
    rig.animation_data_create()
    rig.animation_data.action = act
    keyed = 0
    for b, ent in chans.items():
        frames = np.array(ent["frames"], dtype=float) + 1.0
        Q = np.array(ent["q"])
        for i in range(1, len(Q)):
            if np.dot(Q[i], Q[i - 1]) < 0:
                Q[i] = -Q[i]
        Lc = np.array(ent["loc"])
        for path, data in (("rotation_quaternion", Q), ("location", Lc)):
            for idx in range(data.shape[1]):
                col = data[:, idx]
                if np.ptp(col) < 1e-7 and not (path == "rotation_quaternion" and False):
                    # constant channel: still key when it differs from the rest value
                    rest_val = (1.0 if (path == "rotation_quaternion" and idx == 0) else 0.0)
                    if abs(col[0] - rest_val) < 1e-7:
                        continue
                fc = act.fcurves.new(data_path=f'pose.bones["{b}"].{path}', index=idx, action_group=b)
                fc.keyframe_points.add(len(col))
                co = np.empty(len(col) * 2)
                co[0::2], co[1::2] = frames, col
                fc.keyframe_points.foreach_set("co", co)
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
                fc.update()
                keyed += 1
    scn = bpy.context.scene
    scn.render.fps = FPS
    scn.frame_start, scn.frame_end = 1, n
    if getattr(act, "slots", None) and rig.animation_data.action_slot is None:
        rig.animation_data.action_slot = act.slots[0]
    slot = getattr(rig.animation_data, "action_slot", None)
    print("ANIMDEBUG", act.name, "slots:", [s.name_display for s in getattr(act, "slots", [])], "bound slot:", slot, "fcurves:", len(act.fcurves), "range", tuple(act.frame_range))
    return {"frames": n, "animated_bones": len(chans), "fcurves": keyed}


def verify_against_blender(rig, t=1.9):
    """Max distance between my FK and Blender's evaluated pose for a few key bones (meters)."""
    P = Pose(rig)
    rest_ankle = {sd: SK.BONES[f"shin.{sd}"]["tail"].copy() for sd in "LR"}
    main_pose(P, t, rest_ankle)
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = P.q.get(pb.name, Quaternion())
        pb.location = P.loc.get(pb.name, V())
    bpy.context.view_layer.update()
    worst = 0.0
    for b in ("head", "hand.L", "hand.R", "foot.L", "index3.L", "thigh.R"):
        mine = P.pose(b).translation
        theirs = (rig.matrix_world @ rig.pose.bones[b].matrix).translation
        worst = max(worst, (mine - theirs).length)
    hand = (rig.matrix_world @ rig.pose.bones["hand.L"].matrix).translation  # as evaluated by Blender
    brow = P.pose("head") @ (P.rest["head"].inverted() @ V((0.19, -0.055, 1.45)))
    for pb in rig.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    return worst, (hand - brow).length
