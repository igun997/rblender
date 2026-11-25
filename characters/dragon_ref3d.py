"""Practice against a 3D target: model a dragon head from MEASUREMENTS of a skinned dragon, then measure and correct.

    python3 characters/dragon_ref/prep_glb.py refs3d/red_dragon.glb refs3d/red_dragon_head.npz
    rblender run characters/dragon_ref3d.py -- [refs3d/red_dragon_head.npz] [--fit N] [--no-render]

Target: "Red Dragon" by GadgetHamster (Sketchfab, CC-BY-4.0). The glb and npz stay out of the repo. The target is only
ever MEASURED; nothing of it is copied into the model:
  stage 1  loft   - stations (top, bottom, half-width, superellipse exponent from the cross-section area) cut from the
                    target's skin-weight-segmented parts; horns are tubes along a principal curve; eyes are spheres.
  stage 2  fit    - Regularized Kelvinlet pulls (handle spacing 2 cm) toward the closest target point, best iterate kept.
Every part is a HALF mesh plus a Mirror modifier. Scores are measured against the target, in metres:
  accuracy     model -> target surface distance (is what I built on the target?)
  completeness target -> model surface distance (is the target covered?)  teeth, tongue, frills, nose horn are NOT built
  IoU          side / front / top silhouettes, absolute pixel mapping, no registration
Thresholds were fixed before the first run (fractions of the head length L).
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
NPZ = next((a for a in opts if a.endswith(".npz")), "refs3d/red_dragon_head.npz")
FIT_ITERS = int(opts[opts.index("--fit") + 1]) if "--fit" in opts else 3
FIT_SCHEDULE = [(0.05, FIT_ITERS), (0.035, FIT_ITERS), (0.025, FIT_ITERS)]  # coarse to fine handle spacing (m)
FIT_DIRECTION = opts[opts.index("--direction") + 1] if "--direction" in opts else "closest"
PALETTES = opts[opts.index("--palettes") + 1].split(",") if "--palettes" in opts else ["ember", "emerald", "frost", "obsidian"]
# linear RGB. grade: lift (shadows) / gamma (mids) / gain (highlights) per channel, saturation, contrast
PALETTE_DEFS = {
    "ember": dict(top=(0.06, 0.008, 0.004), side=(0.30, 0.035, 0.012), belly=(0.80, 0.40, 0.10), grout=(0.015, 0.003, 0.002),
                  glow=0.30, glow_color=(1.0, 0.25, 0.02), rough_plate=0.35, rough_grout=0.8, tint=0.35, mottle=0.30,
                  horn_root=(0.05, 0.04, 0.035), horn_mid=(0.25, 0.19, 0.14), horn_tip=(0.75, 0.65, 0.50), eye=(1.0, 0.75, 0.08),
                  lights=dict(Key=(1.0, 0.82, 0.65), Rim=(1.0, 0.50, 0.30)),
                  grade=dict(lift=(0.015, 0.0, -0.01), gamma=(1.0, 0.97, 0.93), gain=(1.08, 0.98, 0.88), saturation=1.15, contrast=0.10)),
    "emerald": dict(top=(0.012, 0.045, 0.03), side=(0.045, 0.15, 0.06), belly=(0.42, 0.33, 0.10), grout=(0.008, 0.02, 0.012),
                    glow=0.0, rough_plate=0.38, rough_grout=0.8, tint=0.40, mottle=0.30,
                    horn_root=(0.06, 0.05, 0.035), horn_mid=(0.28, 0.24, 0.16), horn_tip=(0.72, 0.68, 0.52), eye=(1.0, 0.55, 0.05),
                    lights=dict(Key=(1.0, 0.93, 0.80), Rim=(0.70, 0.90, 1.0)),
                    grade=dict(lift=(-0.01, 0.01, 0.02), gamma=(1.0, 1.0, 1.0), gain=(1.05, 1.0, 0.90), saturation=1.10, contrast=0.08)),
    "frost": dict(top=(0.08, 0.17, 0.30), side=(0.25, 0.45, 0.70), belly=(0.80, 0.90, 0.97), grout=(0.02, 0.05, 0.10),
                  glow=0.25, glow_color=(0.3, 0.7, 1.0), rough_plate=0.22, rough_grout=0.7, tint=0.25, mottle=0.25,
                  horn_root=(0.30, 0.45, 0.55), horn_mid=(0.55, 0.75, 0.88), horn_tip=(0.90, 0.97, 1.0), eye=(0.4, 0.9, 1.0),
                  lights=dict(Key=(0.85, 0.92, 1.0), Rim=(0.6, 0.8, 1.0)),
                  grade=dict(lift=(0.0, 0.01, 0.03), gamma=(1.0, 1.0, 1.02), gain=(0.90, 1.0, 1.10), saturation=0.95, contrast=0.05)),
    "obsidian": dict(top=(0.006, 0.006, 0.008), side=(0.02, 0.02, 0.03), belly=(0.10, 0.06, 0.04), grout=(0.30, 0.10, 0.02),
                     glow=0.80, glow_color=(1.0, 0.35, 0.05), rough_plate=0.22, rough_grout=0.7, tint=0.20, mottle=0.20,
                     horn_root=(0.02, 0.02, 0.02), horn_mid=(0.08, 0.05, 0.04), horn_tip=(0.50, 0.35, 0.15), eye=(1.0, 0.85, 0.2),
                     lights=dict(Key=(1.0, 0.90, 0.80), Rim=(1.0, 0.45, 0.20)),
                     grade=dict(lift=(0.0, 0.0, 0.0), gamma=(1.0, 0.98, 0.95), gain=(1.12, 1.0, 0.92), saturation=1.10, contrast=0.15)),
}
NORENDER = "--no-render" in opts
S = 0.25          # metres per glb unit (puts the head at about 0.6 m)
PPM = 700         # silhouette pixels per metre
T0 = time.time()
rep = prec.Report()
LOG = {}
bpy.context.preferences.edit.use_global_undo = False
scene = rb.clean()
scene.render.film_transparent = False


def lap(name):
    LOG.setdefault("seconds", {})[name] = round(time.time() - T0, 1)


# ------------------------------------------------------------------------------------------------ target
def find(path):
    for c in (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{path}: run characters/dragon_ref/prep_glb.py first (from the project root)")


D = np.load(find(NPZ))
LAB = [str(x) for x in D["labels"]]


def blender_frame(P):
    """glb (x lateral, y up, z back; nose at -z) -> Blender (x lateral, y back, z up; nose at -y), metres."""
    return np.column_stack([P[:, 0], P[:, 2], P[:, 1]]) * S


def tri_label(T, L):
    a, b, c = L[T[:, 0]], L[T[:, 1]], L[T[:, 2]]
    return np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))


def sub_mesh(P, T, L, names):
    ids = [LAB.index(n) for n in names]
    Tk = T[np.isin(tri_label(T, L), ids)]
    used = np.unique(Tk)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    return P[used], remap[Tk]


mP, mL = D["mesh_P"], D["mesh_L"]
mT = D["mesh_T"]
ORIGIN = (lambda B: (B.max(0) + B.min(0)) / 2)(blender_frame(mP))  # head centred at the origin


def tgt(name, P, T):
    return prec.target_mesh(name, blender_frame(P) - ORIGIN, T[:, ::-1])  # the axis swap flips the winding


t_up = tgt("T_upper", *sub_mesh(mP, mT, mL, ["upper"]))
t_jaw = tgt("T_jaw", *sub_mesh(mP, mT, mL, ["jaw"]))
t_horn = tgt("T_horn", *sub_mesh(mP, mT, mL, ["horn.L", "horn.R"]))
t_tongue = tgt("T_tongue", *sub_mesh(mP, mT, mL, ["tongue"]))
t_teeth = tgt("T_teeth", D["teeth_P"], D["teeth_T"])
t_eyes = tgt("T_eyes", D["eyes_P"], D["eyes_T"])
TARGETS = [t_up, t_jaw, t_horn, t_tongue, t_teeth, t_eyes]
COMPARE = [t_up, t_jaw, t_horn, t_eyes]  # what the model sets out to build
allB = np.vstack([prec.coords(o) for o in (t_up, t_jaw, t_horn)])
LEN = float(allB[:, 1].max() - allB[:, 1].min())  # head length L (nose to the back cut)
print(f"TARGET head length L = {LEN * 1000:.1f} mm, size {[round(float(x) * 1000, 1) for x in allB.max(0) - allB.min(0)]} mm", flush=True)
lap("target")


# ------------------------------------------------------------------------------------------------ measuring stations
def dilate(m, r):
    o = m.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            o |= np.roll(np.roll(m, dx, 0), dy, 1)
    return o


def filled_area(pts, cell=0.004, r=3):
    """Area of the region a point cloud outlines: occupancy grid, morphological closing (gaps up to ~2r cells), holes
    filled by flood from outside. Unlike a convex hull it keeps the valleys between spikes."""
    lo = pts.min(0) - (r + 3) * cell
    ij = np.floor((pts - lo) / cell).astype(int)
    g = np.zeros(ij.max(0) + r + 4, bool)
    g[ij[:, 0], ij[:, 1]] = True
    closed = ~dilate(~dilate(g, r), r)
    out = np.zeros_like(closed)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = True
    out &= ~closed
    while True:
        nxt = dilate(out, 1) & ~closed
        if (nxt == out).all():
            break
        out = nxt
    return float((~out).sum()) * cell * cell


def super_ratio(n):
    """Area of a superellipse over the area of its bounding rectangle."""
    return math.gamma(1 + 1 / n) ** 2 / math.gamma(1 + 2 / n)


def exponent_for(ratio):
    lo, hi = 1.6, 12.0
    ratio = min(max(ratio, super_ratio(lo)), super_ratio(hi))
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if super_ratio(mid) < ratio else (lo, mid)
    return (lo + hi) / 2


def slab_stations(Pl, n_st, half_slab):
    """Pl columns: lateral x, along s, vertical b. Rows of (s, half-width, b top, b bottom, exponent) cut from slabs of the
    point cloud; the exponent makes the superellipse area equal the slab's cross-section area."""
    ss = np.linspace(Pl[:, 1].min() + half_slab * 0.5, Pl[:, 1].max() - half_slab * 0.5, n_st)
    rows = []
    for s in ss:
        p = Pl[np.abs(Pl[:, 1] - s) <= half_slab]
        if len(p) < 8:
            rows.append([s, np.nan, np.nan, np.nan, np.nan])
            continue
        bt, bb = np.percentile(p[:, 2], [97, 3])
        w = np.percentile(np.abs(p[:, 0]), 95)
        keep = (p[:, 2] <= bt) & (p[:, 2] >= bb) & (np.abs(p[:, 0]) <= w)
        q = np.column_stack([np.abs(p[keep, 0]), p[keep, 2]])
        area = filled_area(np.vstack([q, q * [-1, 1]]))
        ratio = area / max(4 * w * (bt - bb) / 2, 1e-12)
        rows.append([s, w, bt, bb, exponent_for(ratio)])
    A = np.array(rows)
    for k in range(1, 5):
        ok = ~np.isnan(A[:, k])
        A[:, k] = np.interp(A[:, 0], A[ok, 0], A[ok, k])
        A[1:-1, k] = np.median(np.stack([A[:-2, k], A[1:-1, k], A[2:, k]]), axis=0)  # 3-tap median kills spikes
    return A


def to_frame(ob, origin, ut, n_vec):
    """Local loft coordinates (x, s, b) -> world: y = oy + s*ut_y + b*n_y, z = oz + s*ut_z + b*n_z."""
    P = prec.coords(ob)
    W = P.copy()
    W[:, 1] = origin[0] + P[:, 1] * ut[0] + P[:, 2] * n_vec[0]
    W[:, 2] = origin[1] + P[:, 1] * ut[1] + P[:, 2] * n_vec[1]
    prec.set_coords(ob, W)


# ------------------------------------------------------------------------------------------------ build
clay = prec.simple_material("Clay", (0.62, 0.60, 0.57), rough=0.6, spec=0.3)
horn_mat = prec.simple_material("Horn", (0.50, 0.42, 0.33), rough=0.45, spec=0.4)
eye_mat = prec.simple_material("Eye", (0.9, 0.55, 0.05), rough=0.2, spec=0.8)
tgt_mat = prec.simple_material("Target", (0.35, 0.50, 0.72), rough=0.7, spec=0.2)
teeth_mat = prec.simple_material("TargetTeeth", (0.95, 0.92, 0.80), rough=0.5, spec=0.3)
for o in TARGETS:
    o.data.materials.append(teeth_mat if o is t_teeth else tgt_mat)

HALF_SLAB = 0.011

# --- skull: stations along y, rings in the x-z plane
U = prec.coords(t_up)
ST_SKULL = slab_stations(np.column_stack([U[:, 0], U[:, 1], U[:, 2]]), 60, HALF_SLAB)
skull = prec.loft_half("Skull", ST_SKULL, 110, 56, clay)

# --- jaw: its own frame (the target jaw is open), stations along the principal axis
J = prec.coords(t_jaw)
cj = J[:, 1:].mean(0)
ev, evec = np.linalg.eigh(np.cov((J[:, 1:] - cj).T))
ut = evec[:, 1] * (1 if evec[0, 1] > 0 else -1)          # unit (y, z), pointing from the tip towards the hinge
nv = np.array([-ut[1], ut[0]])                            # perpendicular, z component = ut_y > 0
s_all = (J[:, 1:] - cj) @ ut
b_all = (J[:, 1:] - cj) @ nv
origin_j = cj + s_all.min() * ut
Jl = np.column_stack([J[:, 0], s_all - s_all.min(), b_all])
ST_JAW = slab_stations(Jl, 44, HALF_SLAB)
jaw = prec.loft_half("Jaw", ST_JAW, 90, 48, clay)
to_frame(jaw, origin_j, ut, nv)
LOG["jaw_axis_deg_below_horizontal"] = float(math.degrees(math.atan2(-ut[1], ut[0])))


# --- horns: principal curve through the shell points, radius = rms distance to the curve
def principal_curve(P, K=18, iters=8):
    K = max(3, min(K, len(P) // 4))
    c = P.mean(0)
    ax = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ ax
    edges = np.quantile(t, np.linspace(0, 1, K + 1))
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, K - 1)
    C = np.array([P[idx == k].mean(0) for k in range(K)])
    for _ in range(iters):
        idx = np.linalg.norm(P[:, None] - C[None], axis=2).argmin(1)
        for k in range(K):
            if (idx == k).sum() >= 3:
                C[k] = P[idx == k].mean(0)
        C[1:-1] = 0.5 * C[1:-1] + 0.25 * (C[:-2] + C[2:])
    idx = np.linalg.norm(P[:, None] - C[None], axis=2).argmin(1)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    R = np.zeros(K)
    for k in range(K):
        v = P[idx == k] - C[k]
        if len(v) >= 3:
            perp = v - (v @ T[k])[:, None] * T[k]
            R[k] = math.sqrt((perp ** 2).sum(1).mean())
    ok = R > 0
    if not ok.any():  # a cluster too small to measure a radius: use half its spread
        return C, np.full(K, max(0.5 * float(np.linalg.svd(P - P.mean(0), compute_uv=False)[-1] / math.sqrt(len(P))), 0.002))
    R = np.interp(np.arange(K), np.nonzero(ok)[0], R[ok])
    return C, R  # unoriented, not tapered: the caller decides which end is the root


Hp = prec.coords(t_horn)
Cc, Rr = principal_curve(Hp[Hp[:, 0] > 0])
if Cc[0, 0] > Cc[-1, 0]:  # a horn grows outward: the root has the smaller |x|
    Cc, Rr = Cc[::-1], Rr[::-1]
Rr = Rr.copy()
Rr[-1] *= 0.25  # the tip is a point, not a flat disc
horn = prec.tube("Horn", [tuple(p) for p in Cc], list(Rr), [horn_mat], n=28, cap_start=True, cap_end=True)
prec.add_mirror(horn, merge=False, clip=False, mirror_u=False)

# --- eyes: sphere at the centroid, radius from the mean distance
E = prec.coords(t_eyes)
Er = E[E[:, 0] > 0]
ec = Er.mean(0)
er = float(np.linalg.norm(Er - ec, axis=1).mean())
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=er)
bmesh.ops.translate(bm, vec=tuple(ec), verts=bm.verts[:])
eye = prec.new_object("Eye", bm, [eye_mat])
prec.add_mirror(eye, merge=False, clip=False, mirror_u=False)
MODEL = [skull, jaw, horn, eye]
SPIKES = []
lap("loft")


# ------------------------------------------------------------------------------------------------ measure
def union_dist(P, objs):
    """Distance from each point to the nearest surface among several objects."""
    return np.min([prec.nearest_points(P, prec.bvh_of(o))[1] for o in objs], axis=0)


def stats(d):
    return dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)), max=float(d.max()))


def model_samples():
    """Surface samples of each model part. The back cap of the skull and the jaw, and the buried root of the horn, close a
    cut I made in the target (it is open there), so they are not part of what is being compared."""
    out = {}
    for name, ob in (("skull", skull), ("jaw", jaw), ("horn", horn), ("eye", eye)):
        P = prec.sample_surface(ob, 400 if name == "eye" else 3000, 7)
        if name == "skull":
            P = P[P[:, 1] < ST_SKULL[-1, 0] - 0.012]
        elif name == "jaw":
            P = P[((P[:, 1:] - origin_j) @ ut) < ST_JAW[-1, 0] - 0.012]
        elif name == "horn":
            Q = np.column_stack([np.abs(P[:, 0]), P[:, 1], P[:, 2]])
            P = P[np.linalg.norm(Q - Cc[0], axis=1) > 1.2 * Rr[0]]
        out[name] = P
    if SPIKES:
        pts = []
        for sp in SPIKES:
            P = prec.sample_surface(sp["obj"], 250, 7)
            Q = np.column_stack([np.abs(P[:, 0]) if sp["mirrored"] else P[:, 0], P[:, 1], P[:, 2]])
            pts.append(P[np.linalg.norm(Q - sp["root"], axis=1) > 1.6 * sp["r0"]])
        out["spikes"] = np.vstack(pts)
    return out


def part_scores(tag):
    out = {}
    ms = model_samples()
    d_all = union_dist(np.vstack(list(ms.values())), COMPARE)
    at = 0
    for k, P in ms.items():
        out[k + "_accuracy"] = stats(d_all[at:at + len(P)])
        at += len(P)
    ts = {"upper": t_up, "jaw": t_jaw, "horn": t_horn, "eyes": t_eyes}
    dts = {}
    for k, o in ts.items():
        dts[k] = union_dist(prec.sample_surface(o, 400 if k == "eyes" else 3000, 9), MODEL)
        out[k + "_completeness"] = stats(dts[k])
    d_tm = np.concatenate(list(dts.values()))
    out["whole"] = dict(accuracy=stats(d_all), completeness=stats(d_tm), chamfer_mean=float((d_all.mean() + d_tm.mean()) / 2))
    LOG.setdefault("scores", {})[tag] = out
    return out




def view_masks(objs_model, objs_target):
    """IoU of the three silhouettes plus the masks (absolute mapping, no registration)."""
    tb = np.vstack([prec.coords(o) for o in COMPARE])
    lo, hi = tb.min(0), tb.max(0)
    c = (lo + hi) / 2
    pad = 40
    sizes = {"side": (hi[1] - lo[1], hi[2] - lo[2]), "front": (hi[0] - lo[0], hi[2] - lo[2]), "top": (hi[0] - lo[0], hi[1] - lo[1])}
    cen = {"side": (c[1], c[2]), "front": (c[0], c[2]), "top": (c[0], c[1])}
    out = {}
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "_ref3d_sil.png")
    for v in ("side", "front", "top"):
        res = (int(sizes[v][0] * PPM) + 2 * pad, int(sizes[v][1] * PPM) + 2 * pad)
        a = prec.silhouette(objs_model, cen[v], PPM, res, tmp, view=v)
        b = prec.silhouette(objs_target, cen[v], PPM, res, tmp, view=v)
        out[v] = (prec.iou(a, b), a, b)
    return out


def scores_row(tag):
    sc = part_scores(tag)
    vm = view_masks(MODEL, COMPARE)
    LOG.setdefault("iou", {})[tag] = {k: v[0] for k, v in vm.items()}
    w = sc["whole"]
    print(f"STAGE {tag}: accuracy mean {w['accuracy']['mean'] * 1000:.2f} mm p95 {w['accuracy']['p95'] * 1000:.2f} mm | "
          f"completeness mean {w['completeness']['mean'] * 1000:.2f} mm p95 {w['completeness']['p95'] * 1000:.2f} mm | "
          f"IoU side {vm['side'][0]:.3f} front {vm['front'][0]:.3f} top {vm['top'][0]:.3f}", flush=True)
    for k in ("skull", "jaw", "horn", "eye", "spikes"):
        if k + "_accuracy" in sc:
            print(f"   model {k:6s} accuracy mean {sc[k + '_accuracy']['mean'] * 1000:6.2f} mm  p95 {sc[k + '_accuracy']['p95'] * 1000:6.2f} mm", flush=True)
    for k in ("upper", "jaw", "horn", "eyes"):
        print(f"   target {k:5s} completeness mean {sc[k + '_completeness']['mean'] * 1000:6.2f} mm  p95 {sc[k + '_completeness']['p95'] * 1000:6.2f} mm", flush=True)
    return sc, vm


sc1, vm1 = scores_row("loft")
lap("score_loft")

# ------------------------------------------------------------------------------------------------ correct
fit_log = {}
ROUGH = {"target": {"skull": prec.roughness(t_up), "jaw": prec.roughness(t_jaw)},
         "loft": {"skull": prec.roughness(skull), "jaw": prec.roughness(jaw)}}
for name, ob, tt in (("skull", skull, t_up), ("jaw", jaw, t_jaw)):  # the horn tube is smooth by construction: fitting it twisted it
    t_fit = time.time()
    fit_log[name] = prec.fit_coarse_to_fine(ob, prec.bvh_of(tt), FIT_SCHEDULE, max_step=0.05, gain=0.8, eps_k=2.0,
                                            direction=FIT_DIRECTION, max_roughness=1.5 * prec.roughness(tt))
    print(f"FITTIME {name} {time.time() - t_fit:.1f}s", flush=True)
    print("FIT", name, json.dumps([{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in fit_log[name]]), flush=True)
LOG["fit"] = fit_log
ROUGH["fit"] = {"skull": prec.roughness(skull), "jaw": prec.roughness(jaw)}
LOG["roughness_deg"] = ROUGH
print("ROUGHNESS", json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in ROUGH.items()}), flush=True)
lap("fit")
sc2, vm2 = scores_row("fit")
lap("score_fit")


# ------------------------------------------------------------------------------------------------ residual features
def residual_clusters(obj, thresh=0.012, min_verts=12):
    """Connected groups of target vertices that are still further than `thresh` from the model."""
    P = prec.coords(obj)
    bad = union_dist(P, MODEL) > thresh
    Ed = np.empty(len(obj.data.edges) * 2, np.int32)
    obj.data.edges.foreach_get("vertices", Ed)
    parent = np.arange(len(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in Ed.reshape(-1, 2):
        if bad[a] and bad[b]:
            ra, rb_ = find(a), find(b)
            if ra != rb_:
                parent[ra] = rb_
    roots = np.array([find(i) for i in range(len(P))])
    out = []
    for r in np.unique(roots[bad]):
        pts = P[(roots == r) & bad]
        if len(pts) >= min_verts:
            sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
            if sv[0] >= 1.8 * max(sv[1], 1e-9):  # elongated: a spike, not a missing patch
                out.append(pts)
    return sorted(out, key=len, reverse=True)


def add_spike(i, Pc):
    mirrored = abs(Pc[:, 0].mean()) > 0.02
    if mirrored and Pc[:, 0].mean() < 0:
        return  # the x>0 twin is built and mirrored
    C, R = principal_curve(Pc, K=8, iters=6)
    trees = [prec.bvh_of(o) for o in MODEL]
    near = lambda p: min(prec.nearest_points(p[None], t)[1][0] for t in trees)
    if near(C[-1]) < near(C[0]):  # the root is the end that touches the model
        C, R = C[::-1], R[::-1]
    if near(C[0]) > 0.015:  # not attached: a floating stick is not a feature
        return
    R = np.maximum(R, 0.0015)
    R[-1] *= 0.25
    t0 = C[1] - C[0]
    t0 /= np.linalg.norm(t0)
    C = np.vstack([C[0] - t0 * R[0] * 1.5, C])  # sink the root into the skin
    R = np.concatenate([[R[0]], R])
    ob = prec.tube(f"Spike{i}", [tuple(p) for p in C], list(R), [horn_mat], n=12)
    if mirrored:
        prec.add_mirror(ob, merge=False, clip=False, mirror_u=False)
    SPIKES.append(dict(obj=ob, root=C[0], r0=float(R[0]), mirrored=mirrored, length=float(np.linalg.norm(C[-1] - C[0])), radius=float(R.max())))
    MODEL.append(ob)


SPIKE_MAX = 16
for tobj in (t_up, t_jaw):
    for Pc in residual_clusters(tobj)[:SPIKE_MAX]:
        add_spike(len(SPIKES), Pc)
print(f"SPIKES built: {len(SPIKES)}", flush=True)
sc3, vm3 = scores_row("spikes")
lap("spikes")

# ------------------------------------------------------------------------------------------------ checks
w = sc3["whole"]
rep.at_most("accuracy", "model -> target mean (0.5% of L)", w["accuracy"]["mean"], 0.005 * LEN * 1000)
rep.at_most("accuracy", "model -> target p95 (2% of L)", w["accuracy"]["p95"], 0.02 * LEN * 1000)
rep.at_most("completeness", "target -> model mean (1% of L)", w["completeness"]["mean"], 0.01 * LEN * 1000)
for v in ("side", "front", "top"):
    rep.at_least("silhouette", f"IoU {v}", vm3[v][0], 0.90)
rep.check("symmetry", "skull mirror is a modifier (exact by construction)", any(m.type == "MIRROR" for m in skull.modifiers))
rep.check("fit", "Kelvinlet fit did not make the mean worse than the loft", sc2["whole"]["accuracy"]["mean"] <= sc1["whole"]["accuracy"]["mean"] * 1.001,
          f"{sc1['whole']['accuracy']['mean'] * 1000:.2f} -> {sc2['whole']['accuracy']['mean'] * 1000:.2f} mm")
rep.at_most("smoothness", "skull mean dihedral angle after the fit (<= 1.5x target)", ROUGH["fit"]["skull"], 1.5 * ROUGH["target"]["skull"], unit="deg", scale=1.0)
rep.at_most("smoothness", "jaw mean dihedral angle after the fit (<= 1.5x target)", ROUGH["fit"]["jaw"], 1.5 * ROUGH["target"]["jaw"], unit="deg", scale=1.0)
rep.check("spikes", "residual spikes did not make the mean worse", sc3["whole"]["accuracy"]["mean"] <= sc2["whole"]["accuracy"]["mean"] * 1.05,
          f"{sc2['whole']['accuracy']['mean'] * 1000:.2f} -> {sc3['whole']['accuracy']['mean'] * 1000:.2f} mm")
print(rep.table())

# overlays: model only = cyan, target only = red
tiles = []
for v in ("side", "front", "top"):
    p = rb.out(f"_ov_{v}.png")
    prec.overlay_png(vm3[v][1], vm3[v][2], p)
    tiles.append(p)
prec.contact_sheet(tiles, rb.out("ref3d_overlay.png"))
lap("overlay")

# ------------------------------------------------------------------------------------------------ render
if not NORENDER:
    sc = bpy.context.scene
    prec.studio(floor_z=float(allB[:, 2].min()) - 0.02, floor_r=3.0)
    prec.area_light("Key", (1.3, -1.6, 0.9), (0, 0, 0), 140, 0.9, (1.0, 0.94, 0.86))
    prec.area_light("Fill", (-1.4, -1.2, 0.2), (0, 0, 0), 50, 1.2, (0.78, 0.86, 1.0))
    prec.area_light("Rim", (-0.2, 1.4, 0.8), (0, 0, 0), 170, 0.7, (0.70, 0.90, 1.0))

    def shot(name, cam, res=(640, 640), samples=32):
        sc.camera = cam
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 5
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.filepath = rb.out(name)
        bpy.ops.render.render(write_still=True)
        return sc.render.filepath

    def show(objs):
        for o in bpy.data.objects:
            if o.type == "MESH" and o.name != "Floor" and not o.name.startswith("_"):
                o.hide_render = o not in objs
    cams = {
        "hero": prec.camera((1.0, -1.35, 0.55), (0, 0.0, 0.0), 75, "Hero"),
        "side": prec.camera((1.7, 0, 0), (0, 0, 0), 75, "SideC"),
        "front": prec.camera((0.0, -1.7, 0.1), (0, 0, 0), 75, "FrontC"),
    }
    outs = []
    for tag, objs in (("model", MODEL), ("target", TARGETS)):
        show(objs)
        for k, cam in cams.items():
            outs.append(shot(f"_r_{tag}_{k}.png", cam))
    show(MODEL)
    prec.contact_sheet(outs[:3], rb.out("ref3d_model.png"))
    prec.contact_sheet(outs[3:], rb.out("ref3d_target.png"))
    lap("render")

    # ---- texturing and colour balance: one procedural material set, light colour and compositor grade per palette
    horn_r = np.linalg.norm(Cc, axis=1)
    skin_parts = [skull, jaw]
    horn_parts = [horn] + [sp["obj"] for sp in SPIKES]
    pal_tiles, first_views = [], []
    for pname in PALETTES:
        pal = PALETTE_DEFS[pname]
        m_skin = prec.scale_skin_material(f"Skin_{pname}", pal)
        m_horn = prec.horn_material(f"Horn_{pname}", pal, r0=float(horn_r.min()), r1=float(horn_r.max()))
        m_eye = prec.glow_eye_material(f"Eye_{pname}", pal["eye"])
        for ob, m in [(o, m_skin) for o in skin_parts] + [(o, m_horn) for o in horn_parts] + [(eye, m_eye)]:
            ob.data.materials.clear()
            ob.data.materials.append(m)
        for lname, col in pal["lights"].items():
            lo = bpy.data.objects.get(lname)
            if lo is not None:
                lo.data.color = col
        prec.set_grade(sc, **pal["grade"])
        pal_tiles.append(shot(f"_p_{pname}.png", cams["hero"], res=(720, 720), samples=40))
        if pname == PALETTES[0]:
            first_views = [pal_tiles[-1], shot(f"_p_{pname}_side.png", cams["side"], res=(720, 720), samples=40),
                           shot(f"_p_{pname}_front.png", cams["front"], res=(720, 720), samples=40)]
    prec.contact_sheet(pal_tiles, rb.out("ref3d_palettes.png"))
    prec.contact_sheet(first_views, rb.out("ref3d_textured_views.png"))
    lap("texture")

# ------------------------------------------------------------------------------------------------ report
spec = dict(unit_scale_m_per_glb_unit=S, head_length_m=LEN, skull_stations=ST_SKULL.tolist(), jaw_stations=ST_JAW.tolist(),
            jaw_frame=dict(origin_yz=origin_j.tolist(), axis_yz=ut.tolist(), normal_yz=nv.tolist()),
            horn_path=Cc.tolist(), horn_radius=Rr.tolist(), eye=dict(center=ec.tolist(), radius=er),
            thresholds=dict(accuracy_mean_frac=0.005, accuracy_p95_frac=0.02, completeness_mean_frac=0.01, iou=0.90),
            spikes=[dict(root=sp["root"].tolist(), root_radius=sp["r0"], length=sp["length"], max_radius=sp["radius"], mirrored=sp["mirrored"]) for sp in SPIKES],
            target="Red Dragon, GadgetHamster, CC-BY-4.0 (https://sketchfab.com/3d-models/red-dragon-77e89ab9b88a444cb51339c2ed3959a3)")
with open(rb.out("ref3d_spec.json"), "w") as fh:
    json.dump(spec, fh, indent=1, default=float)
with open(rb.out("ref3d_report.json"), "w") as fh:
    json.dump({"rows": rep.rows, "ok": rep.ok(), "log": LOG}, fh, indent=1, default=float)
with open(rb.out("ref3d_report.txt"), "w") as fh:
    fh.write(rep.table() + "\n")
rb.report(ref3d={"checks": len(rep.rows), "pass": int(sum(r["ok"] for r in rep.rows)), "ok": bool(rep.ok()),
                 "iou_final": {k: v[0] for k, v in vm3.items()}, "seconds": LOG["seconds"]})
