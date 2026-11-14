"""Hair: scalp cap, ~100 tapered clumps (back curtain, face-framing side locks, bangs), twin tufts,
black ribbon bow, hair clips, earring. Back/side clumps are skinned to bone chains for secondary motion."""
import math
import random

import bmesh
from mathutils import Matrix, Vector as V

from . import layout as L
from . import skeleton as SK
from .common import (OBJECTS, box, catmull, join, lerp_path, loft, loft_bm, new_object, rigid_weights,
                     skin_weights, smoothstep, star)

C = L.HEAD_C
R = L.HEAD_R
CHAINS = {}  # group -> [chain bone-name lists]

SKIRT_ENV = [(1.0, 0.108, 0.084), (0.95, 0.125, 0.098), (0.85, 0.17, 0.14), (0.75, 0.22, 0.19), (0.66, 0.265, 0.235)]


def head_point(d, m=1.0):
    d = V(d).normalized()
    return C + V((d.x * R[0] * m, d.y * R[1] * m, d.z * R[2] * m))


def envelope(z):
    """(rx, ry, yc) of the body/skirt hair must stay outside of at height z (None below the skirt)."""
    if z >= 1.0:
        rx, ry, yc = L.torso_at(z)
        pad = 0.030
        if z > 1.18:
            pad += 0.045 * smoothstep(1.18, 1.26, z)  # shrug thickness
        if z > 1.29:
            pad *= 1 - smoothstep(1.29, 1.34, z)
        return rx + pad, ry + pad, yc
    if z >= 0.66:
        ts = [e[0] for e in SKIRT_ENV]
        import numpy as np
        rx = float(np.interp(z, ts[::-1], [e[1] for e in SKIRT_ENV][::-1]))
        ry = float(np.interp(z, ts[::-1], [e[2] for e in SKIRT_ENV][::-1]))
        return rx + 0.012, ry + 0.012, 0.0
    return None


def push_out(p):
    e = envelope(p.z)
    if e is None:
        return p
    rx, ry, yc = e
    q = V((p.x / rx, (p.y - yc) / ry, 0))
    r = q.length
    if r < 1.0:
        if r < 1e-5:
            q = V((0, 1, 0))
            r = 1.0
        q = q / r
        return V((q.x * rx, q.y * ry + yc, p.z))
    return p


def smooth_pts(pts, passes=2):
    pts = [V(p) for p in pts]
    for _ in range(passes):
        out = [pts[0]]
        for i in range(1, len(pts) - 1):
            out.append(pts[i] * 0.5 + (pts[i - 1] + pts[i + 1]) * 0.25)
        out.append(pts[-1])
        pts = out
    return pts


def profile(s):
    return (0.55 + 0.45 * smoothstep(0.0, 0.22, s)) * (1.0 - 0.93 * smoothstep(0.50, 1.0, s) ** 1.2)


def hang_path(phi, ez, length, rng, waves=1.0, front_drift=0.0, steps=14, amp=0.02, flare=0.035):
    """Cylindrical hang: root on the cranium at azimuth phi (0=back, +90=left, 180=front)."""
    cosE = math.sqrt(max(1 - ez * ez, 0.01))
    d0 = V((math.sin(phi) * cosE, math.cos(phi) * cosE, ez))
    root = head_point(d0, 1.04)
    rho0 = math.hypot(root.x, root.y - C.y)
    ph = rng.uniform(0, math.tau)
    f = rng.uniform(1.1, 1.9) * waves
    pts = []
    for k in range(steps):
        s = k / (steps - 1)
        rho = rho0 + 0.03 * (1 - math.exp(-6 * s)) + flare * smoothstep(0.55, 1.0, s)
        a = phi + front_drift * smoothstep((root.z - 1.34) / length, (root.z - 1.34) / length + 0.3, s) + (amp / rho) * math.sin(math.tau * f * s + ph) * smoothstep(0.05, 0.5, s)
        z = root.z - length * s - 0.02 * math.sin(math.pi * s)
        zlin = root.z - length * s
        pts.append(V((rho * math.sin(a), C.y + rho * math.cos(a), zlin if k else root.z)))
    pts[0] = root
    pts = [pts[0]] + [push_out(p) for p in pts[1:]]
    return smooth_pts(pts, 3)


def clump(name, pts, w, t, phi, mat, uv=True, n=8):
    N = len(pts)
    wid = [w * profile(i / (N - 1)) for i in range(N)]
    thk = [t * profile(i / (N - 1)) for i in range(N)]
    bm = bmesh.new()
    side = V((math.cos(phi), -math.sin(phi), 0))
    loft_bm(bm, pts, wid, thk, n=n, side=side, uv=uv)
    return new_object(name, bm, mat, smooth=True)


def cap(M):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=80, v_segments=48, radius=1.0)
    kill = []
    for f in bm.faces:
        for v in f.verts:
            x, y, z = v.co
            if (y < -0.25 and z < 0.52 and abs(x) < 0.80) or z < -0.12:
                kill.append(f)
                break
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    for v in bm.verts:
        v.co = C + V((v.co.x * R[0] * 1.075, v.co.y * R[1] * 1.075, v.co.z * R[2] * 1.075))
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    ob = new_object("Hair_Cap", bm, M["hair_cap"])
    rigid_weights(ob, "head")
    return ob


def build(M):
    rng = random.Random(11)
    parts_static = [cap(M)]
    clumps = []  # (group, pts, w, t, phi)

    # ---- back curtain: two layers around the back/side of the skull
    layer_specs = [(0.30, 0.060, 56), (0.62, 0.050, 40)]
    idx = 0
    for ez0, wbase, count in layer_specs:
        for k in range(count):
            phi = math.radians(-138 + 276 * (k + rng.uniform(-0.3, 0.3)) / (count - 1))
            ez = ez0 + rng.uniform(-0.12, 0.12)
            length = rng.uniform(0.60, 0.74) if ez0 < 0.5 else rng.uniform(0.52, 0.66)
            pts = hang_path(phi, ez, length, rng, waves=1.0, amp=0.024)
            group = min(7, int((phi + math.radians(138)) / math.radians(276) * 8))
            clumps.append(("back", group, pts, wbase * rng.uniform(0.8, 1.2) / 2, 0.0085, phi))
            idx += 1

    # ---- face-framing side locks hanging to the waist, in front of the shoulders
    for s in (1, -1):
        for k in range(4):
            phi = math.radians(s * (96 + 5 * k))
            ez = 0.12 + 0.08 * k
            pts = hang_path(phi, ez, 0.50 + 0.05 * (k % 2), rng, waves=0.9, front_drift=math.radians(s * (70 - 8 * k)),
                            amp=0.012, flare=0.02)
            clumps.append(("side", 0 if s > 0 else 1, pts, 0.017 + 0.003 * k / 2, 0.0075, phi))

    # ---- cheek locks that hide the cap edge at the temples
    for s in (1, -1):
        for k in range(4):
            phi = math.radians(s * (128 + 7 * k))
            pts = hang_path(phi, 0.30 + 0.06 * k, 0.15 + 0.03 * (k % 2), rng, waves=0.5, amp=0.004, flare=0.006, steps=9)
            clumps.append(("bang", 0, pts, 0.016, 0.006, phi))

    # ---- bangs: sweep from the crown over the forehead with a slight side part
    for k in range(13):
        alpha = math.radians(-58 + 116 * k / 12 + rng.uniform(-3, 3))
        b0, b1 = 0.38, 1.12 + 0.05 * math.cos(alpha * 2)
        pts = []
        steps = 11
        for i in range(steps):
            s = i / (steps - 1)
            beta = b0 + (b1 - b0) * s
            a = alpha * (1 + 0.5 * s) + 0.10 * math.sin(s * 3 + k)
            d = V((math.sin(beta) * math.sin(a), -math.sin(beta) * math.cos(a), math.cos(beta)))
            m = 1.05 + 0.02 * smoothstep(0.6, 1.0, s)
            pts.append(head_point(d, m))
        tip = pts[-1]
        pts.append(tip + V((math.sin(alpha) * 0.012, -0.012, -0.016)))
        clumps.append(("bang", 0, smooth_pts(pts, 1), 0.019, 0.006, alpha + math.pi))

    # ---- bone chains for back clumps (8 groups) and side locks (2 groups)
    def chain_for(prefix, members, nb=4):
        N = len(members[0])
        mean = [sum((m[i] for m in members), V()) / len(members) for i in range(N)]
        mean[0] = mean[0] + V((0, 0, 0.0))
        idxs = [round(i * (N - 1) / nb) for i in range(nb + 1)]
        names = []
        parent = "head"
        for i in range(nb):
            nm = f"{prefix}_{i + 1}"
            SK.add_bone(nm, mean[idxs[i]], mean[idxs[i + 1]], parent)
            names.append(nm)
            parent = nm
        return names

    CHAINS["back"] = []
    for g in range(8):
        mem = [c[2] for c in clumps if c[0] == "back" and c[1] == g]
        if mem:
            CHAINS["back"].append(chain_for(f"hairB{g}", mem, 4))
    CHAINS["side"] = []
    for g in (0, 1):
        mem = [c[2] for c in clumps if c[0] == "side" and c[1] == g]
        CHAINS["side"].append(chain_for(f"hairS{g}", mem, 4))

    objs = []
    for n, (kind, g, pts, w, t, phi) in enumerate(clumps):
        ob = clump(f"Hair_{kind}{n}", pts, w, t, phi, M["hair"])
        if kind == "back":
            names = CHAINS["back"][[i for i, ch in enumerate(CHAINS["back"]) if ch[0] == f"hairB{g}_1"][0]] + ["head"]
            skin_weights(ob, SK.segments(names), names, k=3.0, bias={"head": 0.6})
        elif kind == "side":
            names = CHAINS["side"][g] + ["head"]
            skin_weights(ob, SK.segments(names), names, k=3.0, bias={"head": 0.6})
        else:
            rigid_weights(ob, "head")
        objs.append(ob)

    # ---- twin tufts (cat-ear like pointed locks) with their own bones
    CHAINS["tufts"] = []
    for sd, s in L.SIDES.items():
        alpha, beta = math.radians(38 * s), 0.52
        d = V((math.sin(beta) * math.sin(alpha), -math.sin(beta) * math.cos(alpha), math.cos(beta)))
        base = head_point(d, 1.0)
        out = V((math.sin(alpha) * 0.5, -0.25, 0.85)).normalized()
        tip = base + out * 0.058 + V((s * 0.010, 0, 0))
        mid = base + out * 0.030 + V((s * 0.003, 0, 0.0))
        t = loft(f"Hair_Tuft.{sd}", [base - out * 0.012, mid, tip], [0.030, 0.024, 0.003], [0.022, 0.017, 0.003],
                 mat=M["hair"], n=12, side=(math.cos(alpha), -math.sin(alpha), 0), uv=True)
        nm = SK.add_bone(f"tuft.{sd}", base, tip, "head")
        rigid_weights(t, nm)
        CHAINS["tufts"].append([nm])
        objs.append(t)
    # inner shading ridge of the tufts is omitted on purpose (keeps the mesh clean)

    # ---- ribbon bow on the character's right side (viewer left)
    ra, rb = math.radians(-50), 0.78
    rd = V((math.sin(rb) * math.sin(ra), -math.sin(rb) * math.cos(ra), math.cos(rb)))
    rp = head_point(rd, 1.09)
    nrm = V((math.cos(ra), math.sin(ra) * 0 - 0.4, 0.3)).normalized()
    rib = []
    knot = box("Ribbon_Knot", rp, (0.014, 0.014, 0.014), M["black"], bevel=0.003)
    rib.append(knot)
    up = V((-0.55, 0.0, 0.83)).normalized()
    for k, sgn in enumerate((1, -1)):
        ang = math.radians(30) * sgn
        dirv = (Matrix.Rotation(ang, 3, "Y") @ up)
        tang = dirv.cross(V((0, 1, 0))).normalized()
        loop = []
        for i in range(17):
            a = math.tau * i / 16
            loop.append(rp + dirv * (0.032 + 0.032 * -math.cos(a)) + tang * (0.020 * math.sin(a)))
        rib.append(loft(f"Ribbon_Loop{k}", loop, 0.011, 0.0018, mat=M["black"], n=8, side=dirv, caps=(False, False)))
    # tails
    tail_bones = []
    for k, (dx, dz) in enumerate(((-0.030, -0.085), (0.012, -0.10))):
        tail_pts = [rp + V((0, 0.0, 0.0)), rp + V((dx * 0.5, 0.01, dz * 0.45)), rp + V((dx, 0.018, dz * 0.8)), rp + V((dx * 1.4, 0.03, dz))]
        tail = loft(f"Ribbon_Tail{k}", catmull(tail_pts, 5), [0.011 - 0.003 * (i / 18) for i in range(len(catmull(tail_pts, 5)))], 0.0014,
                    mat=M["black"], n=8, side=(1, 0, 0), caps=(True, True))
        names = [SK.add_bone(f"ribbon{k}_1", tail_pts[0], tail_pts[2], "head"), SK.add_bone(f"ribbon{k}_2", tail_pts[2], tail_pts[3], f"ribbon{k}_1")]
        skin_weights(tail, SK.segments(names + ["head"]), names + ["head"], k=3.0, bias={"head": 0.5})
        rib.append(tail)
        tail_bones.append(names)
    CHAINS["ribbon"] = tail_bones
    for o in rib[:3]:
        rigid_weights(o, "head")
    gem = box("Ribbon_Gem", rp + V((0, -0.011, 0.002)), (0.009, 0.004, 0.009), M["gem"], rot=(0, math.radians(45), 0))
    rigid_weights(gem, "head")

    # ---- hair clips near the right eye + earring
    clips = []
    for i, (dx, dz, rot) in enumerate(((-0.050, 1.512, -0.45), (-0.043, 1.506, 0.35))):
        from .body import HEAD_SURFACE
        p = HEAD_SURFACE.project(V((dx, -0.5, dz)), (0, 1, 0), 0.013)
        c = box(f"Clip{i}", p, (0.040, 0.005, 0.007), M["black"], rot=(0, rot, 0), bevel=0.0012)
        rigid_weights(c, "head")
        clips.append(c)
    from .body import HEAD_SURFACE
    cp = HEAD_SURFACE.project(V((-0.058, -0.5, 1.518)), (0, 1, 0), 0.016)
    cg = box("ClipGem", cp, (0.011, 0.005, 0.011), M["gem"], rot=(0, math.radians(45), 0))
    rigid_weights(cg, "head")
    # tassel earring (character's right ear)
    ep = V((-0.093, 0.008, 1.425))
    e1 = loft("Earring_Chain", [ep, ep + V((-0.002, 0, -0.022))], 0.0013, 0.0013, mat=M["gold"], n=6)
    e2 = box("Earring_Bead", ep + V((-0.002, 0, -0.030)), (0.007, 0.007, 0.014), M["blue"], bevel=0.002)
    for o in (e1, e2):
        rigid_weights(o, "head")

    hair = join("Hair", [parts_static[0]] + objs)
    return hair
