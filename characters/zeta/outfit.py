"""Costume: bodice, pleated skirt + black under-layer, belt, olive shrug, detached sleeves, choker + harness,
asymmetric stockings, thigh straps + holster, platform loafers, long side cape, tablet."""
import math

import bmesh
import numpy as np
from mathutils import Matrix, Vector as V

from . import body as B
from . import layout as L
from . import skeleton as SK
from .common import (OBJECTS, Surface, box, catmull, join, lerp_path, loft, loft_bm, new_object, ribbon_decal,
                     rigid_weights, skin_weights, smoothstep, star)

CHAINS = {}
SKIRT_TOP, SKIRT_HEM = 1.0, 0.655
SKIRT_ENV = [(1.0, 0.108, 0.084), (0.95, 0.125, 0.098), (0.85, 0.17, 0.14), (0.75, 0.22, 0.19), (0.655, 0.265, 0.235)]


def skirt_r(z):
    ts = [e[0] for e in SKIRT_ENV][::-1]
    return (float(np.interp(z, ts, [e[1] for e in SKIRT_ENV][::-1])), float(np.interp(z, ts, [e[2] for e in SKIRT_ENV][::-1])))


def strip_mesh(name, path, half_w, mat, cols=8, curve=0.0, side=(1, 0, 0), uv=True):
    """Flat ribbon/panel with UVs (u across, v along the path, path[0] -> v=0)."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    side = V(side)
    N = len(path)
    rows = []
    for i, p in enumerate(path):
        t = (path[min(i + 1, N - 1)] - path[max(i - 1, 0)]).normalized()
        s = (side - t * side.dot(t)).normalized()
        n = s.cross(t)
        row = []
        for c in range(cols + 1):
            x = (c / cols - 0.5) * 2
            row.append(bm.verts.new(p + s * (half_w[i] * x) + n * (curve * x * x * half_w[i])))
        rows.append(row)
    for i in range(N - 1):
        for c in range(cols):
            f = bm.faces.new((rows[i][c], rows[i][c + 1], rows[i + 1][c + 1], rows[i + 1][c]))
            for loop, (ii, cc) in zip(f.loops, ((i, c), (i, c + 1), (i + 1, c + 1), (i + 1, c))):
                loop[uvl].uv = (cc / cols, ii / (N - 1))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    ob = new_object(name, bm, mat, smooth=True)
    return ob


def torso_shell(name, z0, z1, pad, mat, n=40, count=14, caps=(False, False)):
    zs = np.linspace(z0, z1, count)
    cen, rx, ry = [], [], []
    for z in zs:
        hw, hd, yc = L.torso_at(float(z))
        cen.append((0, yc, float(z)))
        rx.append(hw + pad)
        ry.append(hd + pad)
    return loft(name, cen, rx, ry, mat=mat, n=n, caps=caps)


def bodice(M):
    ob = torso_shell("Dress_Bodice", 0.985, 1.215, 0.0065, M["white"], n=48, count=18)
    names = ["hips", "spine", "chest", "clavicle.L", "clavicle.R"]
    skin_weights(ob, SK.segments(names), names, k=3.0)
    sf = Surface(ob)
    # black harness strap: right shoulder -> left waist, and a V strap from the collar
    strap = ribbon_decal("Dress_Strap", None, [V((-0.085, -0.5, 1.212)), V((-0.02, -0.5, 1.15)), V((0.055, -0.5, 1.05)), V((0.088, -0.5, 1.0))],
                         [0.016, 0.017, 0.017, 0.016], M["leather"], surface=sf, offset=0.0025, steps=30)
    skin_weights(strap, SK.segments(names), names, k=3.0)
    strap2 = ribbon_decal("Dress_Strap2", None, [V((0.07, -0.5, 1.255)), V((0.025, -0.5, 1.21))], [0.011, 0.011], M["leather"],
                          surface=sf, offset=0.0025, steps=8)
    skin_weights(strap2, SK.segments(names), names, k=3.0)
    # white pleat panel (front waist) hints
    return ob


def belt(M):
    zs = [0.992, 1.0, 1.012, 1.02]
    cen, rx, ry = [], [], []
    for z in zs:
        hw, hd, yc = L.torso_at(z)
        cen.append((0, yc, z))
        rx.append(hw + 0.011)
        ry.append(hd + 0.011)
    ob = loft("Dress_Belt", cen, rx, ry, mat=M["leather"], n=48, caps=(True, True))
    names = ["hips", "spine"]
    skin_weights(ob, SK.segments(names), names, k=3.0)
    sf = Surface(ob)
    blue = ribbon_decal("Dress_BeltBlue", None, [V((0.03, -0.5, 1.006)), V((0.10, -0.5, 1.006))], [0.017, 0.017], M["blue"], surface=sf,
                        offset=0.0015, steps=20, direction=(0, 1, 0))
    skin_weights(blue, SK.segments(names), names, k=3.0)
    # silver buckle
    _hw, _hd, _yc = L.torso_at(1.006)
    bk = box("Dress_Buckle", V((-0.012, _yc - _hd - 0.013, 1.006)), (0.026, 0.006, 0.022), M["silver"], bevel=0.0018)
    rigid_weights(bk, "spine")
    # blue strap tail hanging over the skirt
    tail = strip_mesh("Dress_BeltTail", [V((0.075, -0.108, 1.0)), V((0.082, -0.15, 0.90)), V((0.088, -0.215, 0.80)), V((0.09, -0.255, 0.74))],
                      [0.012, 0.012, 0.012, 0.004], M["blue"], cols=2, side=(1, 0, 0))
    rigid_weights(tail, "hips")
    return ob


def skirt(M):
    N, n, P = 20, 144, 36
    zs = np.linspace(SKIRT_TOP + 0.004, SKIRT_HEM, N)
    cen, rx, ry = [], [], []
    for z in zs:
        a, b = skirt_r(float(z))
        cen.append((0, 0, float(z)))
        rx.append(a)
        ry.append(b)

    def pleat(i, th):
        t = (th * P / math.tau) % 1.0
        tri = abs(t - 0.5) * 2.0
        g = i / (N - 1)
        return 1.0 + (0.012 + 0.085 * g ** 1.3) * (tri - 0.5)

    bm = bmesh.new()
    loft_bm(bm, cen, rx, ry, n=n, caps=(False, False), radial=pleat, uv=True)
    ob = new_object("Dress_Skirt", bm, M["white"], smooth=True)
    # black under-layer: shorter circumference, hem dips on the character's right-front
    bm2 = bmesh.new()
    zs2 = np.linspace(SKIRT_TOP - 0.01, 0.80, 14)
    cen2 = [(0, 0, float(z)) for z in zs2]
    rx2 = [skirt_r(float(z))[0] * 0.965 for z in zs2]
    ry2 = [skirt_r(float(z))[1] * 0.965 for z in zs2]
    rings = loft_bm(bm2, cen2, rx2, ry2, n=96, caps=(False, False), uv=True)
    for i, ring in enumerate(rings):
        for j, v in enumerate(ring):
            th = math.tau * j / 96
            d = (th - math.radians(150) + math.pi) % math.tau - math.pi
            drop = 0.215 * math.exp(-(d / 0.62) ** 2)
            v.co.z -= drop * (i / (len(rings) - 1)) ** 1.4
    under = new_object("Dress_Underskirt", bm2, M["black"], smooth=True)
    # skirt sway chains (8 around the hips)
    CHAINS["skirt"] = []
    for k in range(8):
        th = math.tau * k / 8
        pts = []
        for z in (1.0, 0.83, SKIRT_HEM + 0.01):
            a, b = skirt_r(z)
            pts.append(V((a * 0.93 * math.cos(th), -b * 0.93 * math.sin(th), z)))
        n1 = SK.add_bone(f"skirt{k}_1", pts[0], pts[1], "hips")
        n2 = SK.add_bone(f"skirt{k}_2", pts[1], pts[2], n1)
        CHAINS["skirt"].append([n1, n2])
    allb = [b for ch in CHAINS["skirt"] for b in ch]
    for o in (ob, under):
        skin_weights(o, SK.segments(allb + ["hips"]), allb + ["hips"], k=2.5, bias={"hips": 0.5}, maxinf=4)
    # diamond chain panel on the front-left
    sf = Surface(ob)
    dm = ribbon_decal("Dress_DiamondPanel", None, [V((-0.045, -0.5, 0.995)), V((-0.06, -0.5, 0.90)), V((-0.075, -0.5, 0.80)), V((-0.085, -0.5, 0.70))],
                      [0.034, 0.036, 0.038, 0.040], M["diamonds"], surface=sf, offset=0.004, steps=40, across=4)
    skin_weights(dm, SK.segments(allb + ["hips"]), allb + ["hips"], k=2.5, bias={"hips": 0.5})
    return ob


def shrug(M):
    cen, rx, ry = [], [], []
    spec = [(1.325, 0.058, 0.056), (1.30, 0.105, 0.075), (1.265, 0.172, 0.088), (1.225, 0.198, 0.092), (1.185, 0.192, 0.090)]
    for z, a, b in spec:
        cen.append((0, 0.004, z))
        rx.append(a)
        ry.append(b)
    cs = lerp_path(catmull(cen, 6), 18)
    t = np.linspace(0, 1, len(cs))
    rxs = np.interp(t, np.linspace(0, 1, len(spec)), rx)
    rys = np.interp(t, np.linspace(0, 1, len(spec)), ry)
    bm = bmesh.new()
    loft_bm(bm, cs, list(rxs), list(rys), n=56, caps=(False, False), uv=True)
    # open the front into a V-neck
    kill = [f for f in bm.faces if f.calc_center_median().y < -0.02 and abs(f.calc_center_median().x) < 0.045 + 0.35 * (1.33 - f.calc_center_median().z)
            and f.calc_center_median().z > 1.215]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    ob = new_object("Outfit_Shrug", bm, M["olive"], smooth=True)
    names = ["chest", "neck", "clavicle.L", "clavicle.R", "upper_arm.L", "upper_arm.R"]
    skin_weights(ob, SK.segments(names), names, k=3.0)
    # little gold button / emblem
    em = box("Outfit_Emblem", V((0.108, -0.092, 1.215)), (0.016, 0.006, 0.020), M["gold"], bevel=0.0015)
    skin_weights(em, SK.segments(names), names, k=3.0)
    return ob


def sleeves(M):
    for sd, s in L.SIDES.items():
        a = L.arm_pts(s)
        arm_path = lerp_path(catmull([a["clav"].lerp(a["shoulder"], 0.55), a["shoulder"].lerp(a["elbow"], 0.25), a["elbow"], a["wrist"]], 14), 22)
        i0, i1 = 6, 19
        path = arm_path[i0:i1 + 1]
        cnt = len(path)
        prof = [B._profile(B.ARM_PROFILE, (i0 + k) / 21) for k in range(cnt)]
        flare = [0.004 + 0.007 * math.sin(math.pi * k / (cnt - 1)) for k in range(cnt)]
        rx = [p[0] + f for p, f in zip(prof, flare)]
        ry = [p[1] + f for p, f in zip(prof, flare)]
        ob = loft(f"Outfit_Sleeve.{sd}", path, rx, ry, mat=M["white"], n=24, side=(0, 1, 0), caps=(False, False))
        names = ["chest", f"clavicle.{sd}", f"upper_arm.{sd}", f"forearm.{sd}", f"hand.{sd}"]
        skin_weights(ob, SK.segments(names), names, k=3.0)
        # cuff + black strap
        cp = lerp_path(arm_path[18:], 4)
        cuff = loft(f"Outfit_Cuff.{sd}", cp, [0.034, 0.036, 0.036, 0.034], [0.030, 0.032, 0.032, 0.030], mat=M["white2"], n=24, side=(0, 1, 0), caps=(False, False))
        skin_weights(cuff, SK.segments(names), names, k=3.0)
        sp = arm_path[19]
        strap = loft(f"Outfit_CuffStrap.{sd}", [sp - V((0, 0, 0.006)), sp + V((0, 0, 0.006))], 0.0385, 0.0345, mat=M["leather"], n=24, side=(0, 1, 0), caps=(False, False))
        skin_weights(strap, SK.segments(names), names, k=3.0)


def collar(M):
    cz0, cz1 = 1.312, 1.342
    cen, rx, ry = [], [], []
    for z in (cz0, 1.327, cz1):
        hw, hd, yc = L.torso_at(z)
        cen.append((0, yc, z))
        rx.append(hw + 0.005)
        ry.append(hd + 0.005)
    ob = loft("Outfit_Collar", cen, rx, ry, mat=M["leather"], n=32, caps=(False, False))
    rigid_weights(ob, "neck")
    hw, hd, yc = L.torso_at(1.30)
    ring = _torus("Outfit_CollarRing", V((0, -(hd + 0.012) + yc, 1.288)), 0.0115, 0.0024, M["silver"])
    rigid_weights(ring, "chest")
    pend = box("Outfit_CollarTag", V((0, -(hd + 0.014) + yc, 1.262)), (0.016, 0.004, 0.022), M["black"], bevel=0.0012)
    rigid_weights(pend, "chest")
    return ob


def _torus(name, center, major, minor, mat, axis="y"):
    bm = bmesh.new()
    n1, n2 = 24, 8
    verts = []
    for i in range(n1):
        a = math.tau * i / n1
        row = []
        for j in range(n2):
            b = math.tau * j / n2
            r = major + minor * math.cos(b)
            p = V((r * math.cos(a), minor * math.sin(b), r * math.sin(a)))  # ring in XZ plane facing -Y
            row.append(bm.verts.new(V(center) + p))
        verts.append(row)
    for i in range(n1):
        for j in range(n2):
            bm.faces.new((verts[i][j], verts[(i + 1) % n1][j], verts[(i + 1) % n1][(j + 1) % n2], verts[i][(j + 1) % n2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat, smooth=True)


def stockings(M):
    # character's left (+X): black thigh-high; right (-X): fishnet sock + knee strap
    for sd, s in L.SIDES.items():
        p = L.leg_pts(s)
        path = lerp_path(catmull([p["hip"] + V((0, 0, 0.02)), p["knee"], p["ankle"] - V((0, 0, 0.03))], 14), 22)
        prof = [B._profile(B.LEG_PROFILE, k / 21) for k in range(22)]
        names = ["hips", f"thigh.{sd}", f"shin.{sd}", f"foot.{sd}"]
        if sd == "L":
            i0 = 5
            ob = loft("Outfit_Stocking.L", path[i0:], [q[0] + 0.0035 for q in prof[i0:]], [q[1] + 0.0035 for q in prof[i0:]],
                      mat=M["stocking"], n=32, caps=(True, True))
            skin_weights(ob, SK.segments(names), names, k=4.0)
            top = path[i0 + 1]
            band = loft("Outfit_StockingBand.L", [top, top - V((0, 0, 0.016))], prof[i0 + 1][0] + 0.0055, prof[i0 + 1][1] + 0.0055, mat=M["black"], n=32, caps=(False, False))
            skin_weights(band, SK.segments(names), names, k=4.0)
        else:
            i0 = 14
            ob = loft("Outfit_Sock.R", path[i0:], [q[0] + 0.0035 for q in prof[i0:]], [q[1] + 0.0035 for q in prof[i0:]],
                      mat=M["sock"], n=32, caps=(True, True), uv=True)
            skin_weights(ob, SK.segments(names), names, k=4.0)
            kp = path[11]
            kstrap = loft("Outfit_KneeStrap.R", [kp + V((0, 0, 0.012)), kp - V((0, 0, 0.012))], prof[11][0] + 0.0075, prof[11][1] + 0.0075, mat=M["leather"], n=32, caps=(False, False))
            skin_weights(kstrap, SK.segments(names), names, k=4.0)
            buckle = box("Outfit_KneeBuckle.R", V((kp.x - 0.0, kp.y - prof[11][1] - 0.008, kp.z)), (0.016, 0.004, 0.018), M["silver"], bevel=0.001)
            skin_weights(buckle, SK.segments(names), names, k=4.0)


def thigh_gear(M):
    # criss-cross straps and a holster on the left thigh (over the stocking)
    p = L.leg_pts(1.0)
    path = lerp_path(catmull([p["hip"] + V((0, 0, 0.02)), p["knee"], p["ankle"] - V((0, 0, 0.03))], 14), 22)
    names = ["hips", "thigh.L", "shin.L"]
    for k, (idx, tilt) in enumerate(((6, 0.045), (8, -0.045))):
        c = path[idx]
        rx, ry = B._profile(B.LEG_PROFILE, idx / 21)
        pts = []
        for i in range(33):
            a = math.tau * i / 32
            pts.append(V((c.x + (rx + 0.0075) * math.cos(a), c.y + (ry + 0.0075) * math.sin(a), c.z + tilt * math.sin(a))))
        b = loft(f"Outfit_ThighStrap{k}", pts, 0.0075, 0.0022, mat=M["leather"], n=6, side=(0, 0, 1), caps=(False, False))
        skin_weights(b, SK.segments(names), names, k=4.0)
    c = path[7]
    rx, ry = B._profile(B.LEG_PROFILE, 7 / 21)
    holster = box("Outfit_Holster", V((c.x + rx + 0.026, c.y - 0.005, c.z - 0.012)), (0.026, 0.045, 0.085), M["leather"], bevel=0.004)
    rigid_weights(holster, "thigh.L")
    flap = box("Outfit_HolsterFlap", V((c.x + rx + 0.026, c.y - 0.004, c.z + 0.036)), (0.028, 0.047, 0.020), M["black"], bevel=0.003)
    rigid_weights(flap, "thigh.L")
    # dangling strap with an arrow tip off the holster
    tip_c = V((c.x + rx + 0.034, c.y - 0.01, c.z - 0.10))
    cord = loft("Outfit_HolsterCord", [V((c.x + rx + 0.034, c.y - 0.01, c.z - 0.045)), tip_c], 0.0035, 0.0012, mat=M["leather"], n=6, side=(0, 1, 0))
    rigid_weights(cord, "thigh.L")
    tip = star("Outfit_HolsterTip", tip_c - V((0, 0, 0.012)), (0, 1, 0), (0, 0, 1), 0.012, 0.004, M["black"], points=2, thickness=0.002)
    rigid_weights(tip, "thigh.L")


def shoes(M):
    for sd, s in L.SIDES.items():
        p = L.leg_pts(s)
        x = p["ankle"].x
        # platform sole
        ys = np.linspace(0.078, -0.145, 14)
        w = [0.036, 0.041, 0.044, 0.047, 0.049, 0.051, 0.051, 0.051, 0.050, 0.048, 0.045, 0.040, 0.030, 0.012]
        cen = [(x, float(y), 0.0185) for y in ys]
        sole = loft(f"Shoe_Sole.{sd}", cen, w, [0.0185] * len(w), mat=M["leather"], n=24, side=(1, 0, 0))
        # upper
        yu = np.linspace(0.072, -0.128, 14)
        wu = [0.032, 0.036, 0.039, 0.041, 0.043, 0.044, 0.044, 0.043, 0.040, 0.036, 0.030, 0.022, 0.013, 0.004]
        hu = [0.038, 0.050, 0.052, 0.046, 0.040, 0.034, 0.030, 0.027, 0.025, 0.023, 0.020, 0.016, 0.011, 0.004]
        cu = [(x, float(y), 0.040 + hh) for y, hh in zip(yu, hu)]
        up = loft(f"Shoe_Upper.{sd}", cu, wu, hu, mat=M["black"], n=24, side=(1, 0, 0))
        bk = box(f"Shoe_Buckle.{sd}", V((x + s * 0.043, -0.01, 0.095)), (0.005, 0.016, 0.014), M["silver"], bevel=0.001)
        names = [f"foot.{sd}", f"toe.{sd}", f"shin.{sd}"]
        for o in (sole, up, bk):
            skin_weights(o, SK.segments(names), names, k=4.0, bias={f"shin.{sd}": 0.15})


def side_cape(M):
    path = [V((-0.17, -0.012, 0.075)), V((-0.20, -0.040, 0.090)), V((-0.205, -0.035, 0.30)), V((-0.19, -0.015, 0.55)), V((-0.17, -0.002, 0.80)), V((-0.155, 0.0, 1.0))]
    pts = lerp_path(catmull(path, 6), 22)
    hw = [0.005 + 0.062 * smoothstep(0.0, 0.5, i / 21) ** 0.7 for i in range(22)]
    ob = strip_mesh("Outfit_SideCape", pts, hw, M["cape"], cols=6, curve=0.10)
    chain = []
    zs = [1.0, 0.75, 0.45, 0.07]
    prev = "hips"
    for k in range(3):
        a = V((-0.155 - 0.02 * k, -0.002 - 0.01 * k, zs[k]))
        b = V((-0.155 - 0.02 * (k + 1), -0.002 - 0.01 * (k + 1), zs[k + 1]))
        prev = SK.add_bone(f"cape_{k + 1}", a, b, prev)
        chain.append(prev)
    CHAINS["cape"] = [chain]
    skin_weights(ob, SK.segments(chain + ["hips"]), chain + ["hips"], k=2.5, bias={"hips": 0.4})
    # star ornaments near the hem
    st = star("Outfit_StarBig", V((-0.196, -0.056, 0.13)), (1, 0, 0), (0, 0, 1), 0.034, 0.008, M["black"], thickness=0.0016)
    st2 = star("Outfit_StarSmall", V((-0.236, -0.056, 0.058)), (1, 0, 0), (0, 0, 1), 0.020, 0.005, M["gem"], thickness=0.0016)
    rigid_weights(st, chain[2])
    rigid_weights(st2, chain[2])
    for o in (st, st2):
        o.location.y -= 0.0


def tablet(M):
    g = L.hand_geometry(-1.0)
    c = g["palm"][2] + V((0.0, -0.085, 0.02))
    t = box("Prop_Tablet", c, (0.125, 0.010, 0.170), [M["tablet"]], rot=(0, 0, math.radians(-12)), bevel=0.0025)
    rigid_weights(t, "hand.R")


def build(M):
    M["white2"] = M.get("white2") or _white2()
    bodice(M)
    belt(M)
    skirt(M)
    shrug(M)
    sleeves(M)
    collar(M)
    stockings(M)
    thigh_gear(M)
    shoes(M)
    side_cape(M)
    tablet(M)


def _white2():
    from .common import material
    return material("Zeta_CuffGrey", (0.78, 0.80, 0.86), rough=0.7)
