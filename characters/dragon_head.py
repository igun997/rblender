"""Reference-driven dragon head: built from a real reference, measured against it, closed-loop, headless.

    rblender run characters/dragon_head.py -- [--quick] [--no-bake] [--no-render] [--open DEG] [--fit N]

Reference: the side view of the 3D dragon skull in Monika Zagrobelna's "How to Draw a Realistic Dragon Head in 3D
Space" (Envato Tuts+, 2013). Only its binary silhouette and hand-measured landmarks are stored, as numbers, in
characters/dragon_ref/ref_silhouette.json. The pixel->metre mapping is absolute (700 px per metre), so the final
IoU is measured with NO registration fitting.

Pipeline: stations from the reference outline and the scheme's braincase ellipse -> HALF loft + Mirror modifier
(exact symmetry) -> declared surface moves solved with Pixar's Regularized Kelvinlets, looped until the measured
eye/brow values land -> silhouette-driven station correction against the reference -> horn crest from the outline
residual -> teeth placed by computation -> high-poly scales (Worley) -> bake NORMAL/AO/albedo onto the loft UVs ->
glTF-safe PBR -> verify against the spec and the reference.
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
from mathutils import Matrix, Vector as V

opts = rb.args()
QUICK = "--quick" in opts
NOBAKE = "--no-bake" in opts
NORENDER = "--no-render" in opts
OPEN_DEG = float(opts[opts.index("--open") + 1]) if "--open" in opts else 0.0
FIT_ITERS = int(opts[opts.index("--fit") + 1]) if "--fit" in opts else 3
TEX = 1024 if QUICK else 2048
JAW_TEX = 512 if QUICK else 1024
VOX = 0.0055 if QUICK else 0.0032
JAW_VOX = 0.005 if QUICK else 0.0032
T0 = time.time()
timing = {}
SIL_TMP = os.path.join(__import__("tempfile").gettempdir(), "_dragon_sil.png")


def lap(name):
    timing[name] = round(time.time() - T0, 1)


rep = prec.Report()
LOG = {"sculpt_loop": [], "sculpt_inner": [], "fit": []}

# =============================================================================================== REFERENCE
def _ref_path():
    cands = ["characters/dragon_ref/ref_silhouette.json"]
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dragon_ref", "ref_silhouette.json"))
    except NameError:
        pass
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("characters/dragon_ref/ref_silhouette.json (run from the project root)")


REF = json.load(open(_ref_path()))
REF_MASK = prec.load_rle_mask(REF)
RH, RW = REF_MASK.shape
REF_TOP, REF_BOT = prec.outline(REF_MASK)
LM = REF["landmarks_px"]
ELL = REF["scheme_px"]["braincase_ellipse"]

# absolute pixel -> world mapping (no fitting): 700 px/m, nose tip at (y=-0.467, z=0), image faces -Y, +Z up
PPM, Y_N, Z_N = 700.0, -0.467, 0.0
NX, NY = LM["nose_tip"]
CY = Y_N + (RW / 2 - (NX + 0.5)) / PPM
CZ = Z_N + ((NY + 0.5) - RH / 2) / PPM


def wy(xp):
    return CY + (np.asarray(xp, float) + 0.5 - RW / 2) / PPM


def wz(yp):
    return CZ + (RH / 2 - np.asarray(yp, float) - 0.5) / PPM


def _fill(a):
    ok = ~np.isnan(a)
    return np.interp(np.arange(len(a)), np.nonzero(ok)[0], a[ok])


TOP_F, BOT_F = _fill(REF_TOP), _fill(REF_BOT)


def ref_top_at(x):
    return np.interp(x, np.arange(RW), TOP_F)


def ref_bot_at(x):
    return np.interp(x, np.arange(RW), BOT_F)


def sstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def ell_y(x, sign):
    t = 1 - ((np.asarray(x, float) - ELL["cx"]) / ELL["a"]) ** 2
    return ELL["cy"] + sign * ELL["b"] * np.sqrt(np.clip(t, 0, None))


def body_bot_px(x):
    x = np.asarray(x, float)
    w = sstep(280, 310, x)
    flat = np.minimum(307.0, ref_bot_at(x) - 2)
    base = flat * (1 - w) + w * ell_y(x, +1)
    wb = sstep(340, 390, x)  # beyond the jaw hinge the reference bottom is the occipital bulge
    return base * (1 - wb) + wb * ref_bot_at(x)


def body_top_px(x):
    """Braincase/snout top (px): the reference outline, blending into the scheme's braincase ellipse where the
    outline leaves it to become the horn crest; past x=430 the dome closes over the occipital bulge."""
    x = np.asarray(x, float)
    rt = ref_top_at(x)
    t = rt + sstep(245, 300, x) * (ell_y(x, -1) - rt) * (x > 178)
    t430 = float(ref_top_at(430.0) + sstep(245, 300, 430.0) * (ell_y(430.0, -1) - ref_top_at(430.0)))
    h0 = float(body_bot_px(430.0)) - t430
    c = np.clip((x - 430.0) / 18.0, 0, 1)
    return np.where(x > 430.0, body_bot_px(x) - h0 * np.sqrt(np.clip(1 - c ** 2, 0, None)), t)


# =============================================================================================== SPEC
S = dict(
    eye_c=(0.092, float(wy(LM["orbit_center"][0])), float(wz(LM["orbit_center"][1]))), eye_r=0.022,
    eye_protrusion=0.010, brow_h=float((LM["orbit_center"][1] - LM["brow_top"][1]) / PPM),
    hinge=(0.0, float(wy(LM["hinge"][0])), float(wz(LM["hinge"][1]))), open_deg=OPEN_DEG,
    teeth_up=9, teeth_lo=8, canine_up=0.034, canine_lo=0.030,
    snout_tip_y=float(wy(NX)),
)

HX = np.concatenate([np.linspace(NX, 430, 72), [434.0, 438.0, 442.0, 445.0, 447.5]])
JX = np.linspace(36, 395, 60)
HEAD_TOP_PX = body_top_px(HX).astype(float)
HEAD_BOT_PX = body_bot_px(HX).astype(float)
_jt = np.interp(JX, [36, 300, 314, 350], [305.0, 305.0, 246.0, 246.0])
_jb = ref_bot_at(JX)
JAW_BOT_PX = np.maximum(_jb, _jt + 10)
JAW_TOP_PX = np.where(JX <= 350, _jt, (1 - sstep(350, 356, JX)) * 246.0 + sstep(350, 356, JX) * (JAW_BOT_PX - 42))


def make_ST(xs, top_px, bot_px, head=True, head_w=None):
    zt, zb, y = wz(top_px), wz(bot_px), wy(xs)
    h = zt - zb
    if head:
        kw = 1.0 - 0.08 * sstep(180, 300, xs)
        n = 2.6 - 0.3 * sstep(100, 300, xs)
        w = kw * h / 2
    else:
        n = np.full(len(xs), 2.6)
        w = np.interp(y, head_w[0], head_w[1]) * (0.80 + 0.12 * sstep(250, 330, xs))
    return np.column_stack([y, w, zt, zb, n])


def E(name, origin, direction, mm, eps):
    return dict(name=name, origin=origin, dir=direction, mm=mm, eps=eps)


def top_ray(lat, xp):
    return (lat, float(wy(xp)), 0.5), (0, 0, -1)


def side_ray(xp, yp):
    return (0.4, float(wy(xp)), float(wz(yp))), (-1, 0, 0)


def head_edits(p):
    """Declared surface moves (mm along the surface normal at the ray hit); entries with x>0 are mirrored."""
    def e(name, ray, mm, eps):
        return E(name, ray[0], ray[1], mm, eps)
    return [
        e("brow_peak", top_ray(0.070, 235), 10.0, 0.022),
        e("brow_back", top_ray(0.084, 270), 11.0, 0.020),
        e("brow_front", top_ray(0.056, 200), 12.0, 0.018),
        e("eye_socket", side_ray(228, 203), -p["socket"], 0.016),
        e("lower_lid", side_ray(232, 245), 5.0, 0.016),
        e("cheekbone", side_ray(250, 268), 10.0, 0.030),
        e("maxilla", side_ray(130, 275), 6.0, 0.025),
        e("nostril_rim", top_ray(0.026, 80), 6.0, 0.013),
        e("nostril_dent", top_ray(0.027, 55), -6.0, 0.010),
        e("nose_bulb", top_ray(0.0, 28), 6.0, 0.016),
        e("snout_ridge_mid", top_ray(0.0, 140), 9.0, 0.040),
        e("snout_ridge_fore", top_ray(0.0, 100), 12.0, 0.040),
        e("forehead", top_ray(0.0, 215), 7.0, 0.050),
        e("temple", side_ray(330, 185), -8.0, 0.040),
        e("occipital", top_ray(0.0, 410), 12.0, 0.030),
    ]


JAW_EDITS = [
    E("chin", (0.0, float(wy(60)), -0.5), (0, 0, 1), 4.0, 0.015),
    E("jaw_mid", (0.4, float(wy(200)), float(wz(330))), (-1, 0, 0), 4.0, 0.030),
    E("jaw_angle", (0.4, float(wy(345)), float(wz(300))), (-1, 0, 0), 9.0, 0.030),
]


# =============================================================================================== BUILD
def drop(ob):
    me = ob.data
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)


bpy.context.preferences.edit.use_global_undo = False
scene = rb.clean()
scene.render.film_transparent = False
prec.studio(floor_z=-0.14, floor_r=3.0)
prec.area_light("Key", (1.3, -1.6, 0.9), (0, -0.05, 0.08), 140, 0.9, (1.0, 0.94, 0.86))
prec.area_light("Fill", (-1.4, -1.2, 0.2), (0, -0.05, 0.08), 50, 1.2, (0.78, 0.86, 1.0))
prec.area_light("Rim", (-0.2, 1.4, 0.8), (0, 0.0, 0.1), 170, 0.7, (0.70, 0.90, 1.0))
prec.area_light("Under", (0.0, -0.9, -0.5), (0, -0.2, 0.0), 20, 1.0, (1.0, 0.85, 0.7))
clay = prec.simple_material("Clay", (0.30, 0.29, 0.28), rough=0.7, spec=0.3)
tmp_mat = prec.simple_material("BakeTarget", (0.5, 0.5, 0.5))


def sculpt(ob, P0, edits, tol_mm=0.10, max_iter=4, pin_spacing=0.06, pin_eps=0.03, tag="head"):
    """Declared moves on the HALF mesh. Handles, mirrored copies and pins feed one Kelvinlet constraint solve; the
    displacement is applied to the half only and clipped on the mirror plane, exactly like the Mirror modifier."""
    prec.set_coords(ob, P0)
    tree = prec.bvh_of(ob)  # evaluated (mirrored) surface
    M = np.array([-1.0, 1.0, 1.0])
    A = []
    for e in edits:
        loc, nor, _ = prec.ray(tree, V(e["origin"]), V(e["dir"]))
        assert loc is not None, f"edit {e['name']} missed the surface"
        A.append(dict(name=e["name"], p=np.array(loc), n=np.array(nor), mm=e["mm"], eps=e["eps"], mir=abs(e["origin"][0]) > 1e-9))
    pins = prec.pin_points(P0, [a["p"] for a in A] + [a["p"] * M for a in A if a["mir"]],
                           [1.6 * a["eps"] for a in A] + [1.6 * a["eps"] for a in A if a["mir"]], pin_spacing)
    center = np.abs(P0[:, 0]) < 1e-9
    corr = np.zeros(len(A))
    best_err, best_P, best_info = float("inf"), None, None
    for it in range(max_iter):
        hp = [a["p"] for a in A] + [a["p"] * M for a in A if a["mir"]]
        hd = [a["n"] * (a["mm"] + c) / 1000.0 for a, c in zip(A, corr)] + \
             [a["n"] * M * (a["mm"] + c) / 1000.0 for a, c in zip(A, corr) if a["mir"]]
        he = [a["eps"] for a in A] + [a["eps"] for a in A if a["mir"]]
        H = np.vstack([np.array(hp), pins])
        D = np.vstack([np.array(hd), np.zeros((len(pins), 3))])
        eps = np.concatenate([he, np.full(len(pins), pin_eps)])
        F, res, reg, U = prec.solve_stable(H, D, eps, P0, max_disp=0.05)
        U[center, 0] = 0.0
        prec.set_coords(ob, P0 + U)
        tree1 = prec.bvh_of(ob)
        errs = []
        for a in A:
            loc, _n, dist = prec.ray(tree1, V(a["p"]) - V(a["n"]) * 0.012, V(a["n"]), 0.2)
            errs.append(a["mm"] - ((dist - 0.012) * 1000.0 if loc is not None else float("nan")))
        errs = np.nan_to_num(np.array(errs))
        cur = float(np.abs(errs).max())
        info = dict(tag=tag, it=it, anchors=len(A), pins=len(pins), solve_residual_m=res, regularisation=reg, max_err_mm=cur,
                    worst=A[int(np.abs(errs).argmax())]["name"])
        LOG["sculpt_inner"].append(info)
        if cur < best_err:
            best_err, best_P, best_info = cur, prec.coords(ob), dict(info)
        if cur < tol_mm:
            break
        corr += errs
    prec.set_coords(ob, best_P)
    best_info["moves"] = [dict(name=a["name"], requested_mm=a["mm"], eps_m=a["eps"]) for a in A]
    return best_info


params = {"socket": 3.0}
hist = {"socket": []}


def secant(name, target_mm, measured_mm, gain_floor=0.05):
    h = hist[name]
    h.append((params[name], measured_mm))
    slope = 1.0
    if len(h) >= 2 and abs(h[-1][0] - h[-2][0]) > 1e-9:
        s = (h[-1][1] - h[-2][1]) / (h[-1][0] - h[-2][0])
        if s > gain_floor:
            slope = s
    params[name] += (target_mm - measured_mm) / slope


def build_head_and_jaw():
    head = prec.loft_half("Dragon_Head", make_ST(HX, HEAD_TOP_PX, HEAD_BOT_PX), 200, 80, tmp_mat)
    P0 = prec.coords(head)
    info = None
    for k in range(10):
        info = sculpt(head, P0, head_edits(params))
        tree = prec.bvh_of(head)
        loc, _n, _d = prec.ray(tree, V((0.4, S["eye_c"][1], S["eye_c"][2])), V((-1, 0, 0)))
        protr = (S["eye_c"][0] + S["eye_r"]) - loc.x
        LOG["sculpt_loop"].append(dict(iter=len(LOG["sculpt_loop"]), socket_mm=round(params["socket"], 3),
                                       eye_protrusion_mm=round(protr * 1000, 3), inner=info["max_err_mm"]))
        if abs(protr - S["eye_protrusion"]) < 0.0002:
            break
        secant("socket", S["eye_protrusion"] * 1000, protr * 1000)
        params["socket"] = float(np.clip(params["socket"], -30.0, 30.0))
    hst = make_ST(HX, HEAD_TOP_PX, HEAD_BOT_PX)
    jaw = prec.loft_half("Dragon_Jaw", make_ST(JX, JAW_TOP_PX, JAW_BOT_PX, head=False, head_w=(hst[:, 0], hst[:, 1])), 110, 64, tmp_mat)
    jP0 = prec.coords(jaw)
    jinfo = sculpt(jaw, jP0, JAW_EDITS, tag="jaw")
    return head, P0, info, jaw, jP0, jinfo, hst


def smooth_err(e, k=9):
    e = np.where(np.isnan(e), 0.0, e)
    return np.convolve(e, np.ones(k) / k, mode="same")


# ---- silhouette-driven correction: the body outline is pulled onto the reference outline ----------------------
head = jaw = None
for fit in range(FIT_ITERS + 1):
    if head is not None:
        drop(head)
        drop(jaw)
    head, HEAD_P0, head_info, jaw, JAW_P0, jaw_info, HEAD_ST = build_head_and_jaw()
    mask = prec.silhouette([head, jaw], (CY, CZ), PPM, (RW, RH), SIL_TMP)
    top_m, bot_m = prec.outline(mask)
    et = (top_m - REF_TOP)
    eb = (bot_m - REF_BOT)
    ok_t = ~np.isnan(et)
    ok_t[246:] = False  # beyond the brow the top outline is the horn crest, not the body
    ok_b = ~np.isnan(eb)
    ok_b[450:] = False
    ok_b[:int(JX[0]) + 10] = False  # the rounded jaw tip does not cover these columns yet: not an outline error
    LOG["fit"].append(dict(iter=fit, top_err_mean_px=float(np.abs(et[ok_t]).mean()), top_err_max_px=float(np.abs(et[ok_t]).max()),
                           bottom_err_mean_px=float(np.abs(eb[ok_b]).mean()), bottom_err_max_px=float(np.abs(eb[ok_b]).max())))
    print("FITITER " + json.dumps(LOG["fit"][-1]), flush=True)
    if fit == FIT_ITERS:
        break
    sT = np.clip(smooth_err(np.where(ok_t, et, np.nan)), -12, 12)
    sB = np.clip(smooth_err(np.where(ok_b, eb, np.nan)), -12, 12)
    HEAD_TOP_PX -= 0.9 * np.interp(HX, np.arange(RW), sT) * (HX <= 245)
    JAW_BOT_PX -= 0.9 * np.interp(JX, np.arange(RW), sB)
    HEAD_BOT_PX -= 0.9 * np.interp(HX, np.arange(RW), sB) * (HX > 395)
print("FIT " + json.dumps(LOG["fit"]))
print("SCULPT_LOOP " + json.dumps(LOG["sculpt_loop"][-4:]))
HEAD_SCULPTED = prec.coords(head)
lap("loft_sculpt_fit")

blockout = prec.duplicate(head, "Dragon_Blockout")
prec.set_coords(blockout, HEAD_P0)
blockout.data.materials.clear()
blockout.data.materials.append(clay)
blockout.hide_render = True
heat = prec.duplicate(head, "Dragon_Heat")
heat.data.materials.clear()
_ca = heat.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
_ca.data.foreach_set("color", prec.heat_rgba(np.linalg.norm(HEAD_SCULPTED - HEAD_P0, axis=1), 0.016).astype(np.float32).ravel())
heat.data.materials.append(prec.vertex_color_material("Heat", "Col"))
heat.hide_render = True

# ---- horn crest: the part of the reference outline that is not braincase -------------------------------------
head_tree, jaw_tree = prec.bvh_of(head), prec.bvh_of(jaw)
mat_horn = prec.simple_material("Horn", (0.20, 0.17, 0.12), rough=0.45, spec=0.5)
mat_tooth = prec.simple_material("Tooth", (0.86, 0.82, 0.70), rough=0.35, spec=0.5)
mat_tongue = prec.simple_material("Tongue", (0.62, 0.16, 0.20), rough=0.45, spec=0.4)

hxs = np.arange(238.0, 582.0, 4.0)
h_top = ref_top_at(hxs)
b_top = body_top_px(hxs)
EMBED = 6.0  # px sunk into the braincase so the crest is attached, not floating
XB, XT = 432.0, 452.0
t_body = np.where(hxs <= XB, b_top + EMBED - h_top, 0.0)
tB = float(body_top_px(XB) + EMBED - ref_top_at(XB))
tT = float(ref_bot_at(XT) - ref_top_at(XT))  # horn-only: the occipital bulge ends at x=449
t_tail = ref_bot_at(hxs) - h_top
thick = np.where(hxs <= XB, t_body, np.where(hxs >= XT, t_tail, tB + (tT - tB) * (hxs - XB) / (XT - XB)))
keep = thick >= 3.0
hxs, h_top, thick = hxs[keep], h_top[keep], thick[keep]
h_r = np.maximum(thick / 2 / PPM, 0.0012)
h_lat = 0.035 + 0.5 * np.minimum(h_r, 0.040)  # constant over the thick base: a lateral drift tilts the rings and bloats the side view
h_path = np.column_stack([h_lat, wy(hxs), wz(h_top + thick / 2)])
horn = prec.tube("Horn", h_path, h_r, [mat_horn], n=18, cap_start=True, cap_end=True, axis=(0, 1, 0))
prec.add_mirror(horn, merge=False, clip=False, mirror_u=False)
lap("horn")


def cone_bm(bm, root, direction, length, r, back, curve=0.22, n=8):
    d = V(direction).normalized()
    root = V(root)
    ts = [-0.14, 0.0, 0.30, 0.62, 1.0]
    pts = [root + d * length * t + V(back) * length * curve * max(t, 0) ** 2 for t in ts]
    prec.tube_bm(bm, pts, [r, r, 0.78 * r, 0.45 * r, 0.03 * r], n=n, squash=0.85)


def tooth_line(ST, scale, y0, y1, count, offset=0.0):
    ys = np.linspace(y0, y1, 80)
    pts = np.array([[scale * prec.hermite1d(ST[:, 0], ST[:, 1], y), y, 0.0] for y in ys])
    pos, _t = prec.arc_positions(pts, count + 1, margin=0.0)
    return pos[:count] + offset * (pos[1:count + 1] - pos[:count])


def make_teeth(name, tree, ray_dir, ray_z, ST, scale, count, canine_i, canine_len, sizes, y0, y1, offset, up, mat):
    bm = bmesh.new()
    placed = []
    line = tooth_line(ST, scale, y0, y1, count, offset)
    left = []
    for i, p in enumerate(line):
        loc0, nor0, _ = prec.ray(tree, V((p[0], p[1], ray_z)), V(ray_dir))
        assert loc0 is not None, f"{name} tooth {i} missed"
        left.append((loc0, nor0))
    for sgn in (1, -1):
        for i, (loc0, nor0) in enumerate(left):
            loc, nor = V((sgn * loc0.x, loc0.y, loc0.z)), V((sgn * nor0.x, nor0.y, nor0.z))
            length = canine_len if i == canine_i else sizes[min(i, len(sizes) - 1)]
            r = 0.0070 if i == canine_i else 0.0036 + 0.0004 * (i % 3)
            d = V(nor) * 0.7 + V((0, 0, -1 if not up else 1)) * 0.3 + V((sgn * 0.10, -0.02 if i == canine_i else 0.05, 0))
            cone_bm(bm, loc, d, length, r, back=(0, 1, 0), n=8)
            placed.append((tuple(loc), length))
    return prec.new_object(name, bm, [mat]), placed


UP_SIZES = [0.013, 0.012, 0.034, 0.012, 0.014, 0.016, 0.017, 0.017, 0.016]
LO_SIZES = [0.014, 0.030, 0.012, 0.014, 0.015, 0.016, 0.016, 0.015]
JAW_ST = make_ST(JX, JAW_TOP_PX, JAW_BOT_PX, head=False, head_w=(HEAD_ST[:, 0], HEAD_ST[:, 1]))
teeth_up, up_placed = make_teeth("Teeth_Upper", head_tree, (0, 0, 1), -0.30, HEAD_ST, 0.80, S["teeth_up"], 2, S["canine_up"],
                                 UP_SIZES, float(wy(40)), float(wy(225)), 0.0, False, mat_tooth)
teeth_lo, lo_placed = make_teeth("Teeth_Lower", jaw_tree, (0, 0, -1), 0.30, JAW_ST, 0.62, S["teeth_lo"], 1, S["canine_lo"],
                                 LO_SIZES, float(wy(46)), float(wy(215)), 0.5, True, mat_tooth)
t_ys = np.linspace(float(wy(90)), float(wy(335)), 40)
t_pts = []
for y in t_ys:
    loc, _n, _ = prec.ray(jaw_tree, V((0.0, y, 0.3)), V((0, 0, -1)))
    if loc is not None:
        t_pts.append(tuple(loc))
t_pts = np.array(t_pts) + np.array([0, 0, 0.007])
tongue = prec.tube("Tongue", t_pts, np.interp(np.linspace(0, 1, len(t_pts)), [0, 0.7, 1], [0.0035, 0.0065, 0.0075])[::-1], [mat_tongue],
                   n=12, squash=3.4)
lap("teeth")


# ---- eyes with slit pupil, UV-mapped by gaze direction
def eye_texture(size=256):
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1.0)
    u, v = xx - 0.5, yy - 0.5
    r = np.hypot(u, v)
    ang = np.arctan2(v, u)
    img = np.ones((size, size, 3)) * np.array([0.78, 0.66, 0.30])
    iris_r = 0.34
    streak = 0.5 + 0.5 * np.sin(ang * 22 + r * 18)
    iris = np.array([0.95, 0.52, 0.06])[None, None] * (0.65 + 0.35 * streak[..., None])
    iris = iris * (0.55 + 0.9 * np.clip(r / iris_r, 0, 1) ** 1.6)[..., None] * 0.9 + np.array([0.25, 0.14, 0.0]) * 0.25
    ring = np.clip((r - iris_r * 0.9) / (iris_r * 0.1), 0, 1)[..., None]
    iris = iris * (1 - ring) + np.array([0.08, 0.04, 0.0]) * ring
    m_iris = np.clip((iris_r - r) * 120, 0, 1)[..., None]
    img = img * (1 - m_iris) + iris * m_iris
    mp = np.clip((1 - ((u / 0.034) ** 2 + (v / 0.215) ** 2)) * 6, 0, 1)[..., None]
    img = img * (1 - mp) + np.array([0.01, 0.01, 0.01]) * mp
    return prec.image_from_array("dragon_eye", img)


mat_eye = prec.simple_material("Eye", (1, 1, 1), rough=0.12, spec=0.8)
tn = mat_eye.node_tree
tx = tn.nodes.new("ShaderNodeTexImage")
tx.image = eye_texture()
tn.links.new(tx.outputs["Color"], tn.nodes["Principled BSDF"].inputs["Base Color"])
eyes = []
for sgn, sd in ((1, "L"), (-1, "R")):
    c = V((sgn * S["eye_c"][0], S["eye_c"][1], S["eye_c"][2]))
    g = V((sgn * 0.70, -0.70, 0.10)).normalized()
    right = g.cross(V((0, 0, 1))).normalized()
    up = right.cross(g).normalized()
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=20, radius=S["eye_r"])
    uvl = bm.loops.layers.uv.verify()
    for f in bm.faces:
        for loop in f.loops:
            nv = loop.vert.co.normalized()
            loop[uvl].uv = (0.5 + 0.5 * nv.dot(right), 0.5 + 0.5 * nv.dot(up)) if nv.dot(g) > 0 else (0.01, 0.01)
    bmesh.ops.translate(bm, vec=c, verts=bm.verts)
    eyes.append(prec.new_object("Eye." + sd, bm, [mat_eye]))
lap("eyes")

# ---- the Mirror modifiers become real geometry now (symmetric by construction), then the jaw opens
for ob in (head, jaw, horn):
    prec.apply_modifiers(ob)
hx, hy, hz = S["hinge"]
Mopen = Matrix.Translation((hx, hy, hz)) @ Matrix.Rotation(math.radians(S["open_deg"]), 4, "X") @ Matrix.Translation((-hx, -hy, -hz))
JAW_TIP_IDX = int(prec.coords(jaw)[:, 1].argmin())
JAW_TIP_CLOSED = prec.coords(jaw)[JAW_TIP_IDX].copy()
JAW_SURF_BEFORE = prec.bvh_of(jaw)
jaw_group = [jaw, teeth_lo, tongue]


# =============================================================================================== SKIN
def uv_stats(ob, tex):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    uv = bm.loops.layers.uv.active
    dens = []
    for f in bm.faces:
        if len(f.verts) != 4:
            continue
        l = f.loops
        ua = abs((l[1][uv].uv - l[0][uv].uv).cross(l[3][uv].uv - l[0][uv].uv)) + abs((l[1][uv].uv - l[2][uv].uv).cross(l[3][uv].uv - l[2][uv].uv))
        ga = f.calc_area()
        if ga > 1e-9 and ua > 1e-12:
            dens.append(math.sqrt(ua / 2 * tex * tex / ga) / 1000.0)
    bm.free()
    d = np.array(dens)
    return {"px_per_mm_mean": float(d.mean()), "px_per_mm_p5": float(np.percentile(d, 5)), "px_per_mm_p95": float(np.percentile(d, 95))}


def skin_pipeline(low, tag, vox, tex, seed, ao_samples, amp=0.0016, size_far=0.0165, size_near=0.0075):
    hi = prec.duplicate(low, tag + "_Hi")
    hi.data.materials.clear()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    hi.select_set(True)
    bpy.context.view_layer.objects.active = hi
    hi.data.remesh_voxel_size = vox
    hi.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    hi.data.update()
    P = prec.coords(hi)
    N = prec.vertex_normals(hi)
    size = size_near + (size_far - size_near) * prec.smoothstep(float(wy(40)), float(wy(300)), P[:, 1])
    F1, F2, ID = prec.worley(P * np.array([1.0, 0.75, 1.0]), size, seed=seed)
    plate = prec.smoothstep(0.03, 0.30, F2 - F1) * (1 - 0.35 * F1)
    palate = prec.smoothstep(-0.55, -0.8, N[:, 2]) * prec.smoothstep(S["hinge"][1] + 0.02, S["hinge"][1] - 0.02, P[:, 1])
    h = (plate - 0.5) * amp * (0.8 + 0.4 * ID) * (1 - 0.85 * palate)
    hi.data.vertices.foreach_set("co", (P + N * h[:, None]).astype(np.float32).ravel())
    hi.data.update()
    P2 = prec.coords(hi)
    N2 = prec.vertex_normals(hi)
    nz = N2[:, 2]
    top = np.array([0.020, 0.055, 0.040])
    side = np.array([0.055, 0.125, 0.060])
    belly = np.array([0.30, 0.22, 0.075])
    wb = prec.smoothstep(0.15, -0.55, nz)[:, None]
    ws = prec.smoothstep(0.75, 0.15, nz)[:, None]
    col = top[None] * (1 - ws) + side[None] * ws
    col = col * (1 - wb) + belly[None] * wb
    col *= (0.82 + 0.36 * ID)[:, None]
    col *= (0.30 + 0.70 * prec.smoothstep(0.0, 0.25, F2 - F1))[:, None]
    mouth = (prec.smoothstep(-0.55, -0.8, nz) * prec.smoothstep(S["hinge"][1] + 0.02, S["hinge"][1] - 0.02, P2[:, 1]))[:, None]
    col = col * (1 - mouth) + np.array([0.42, 0.075, 0.09])[None] * mouth * (0.7 + 0.3 * ID[:, None])
    rgba = np.concatenate([col, np.ones((len(col), 1))], axis=1)
    ca = hi.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", rgba.astype(np.float32).ravel())
    m = prec.vertex_color_material(tag + "_HiSkin", "Col", rough=0.5)
    hi.data.materials.append(m)
    hi.data.shade_smooth()
    out = {"hi": hi, "hi_tris": sum(len(p.vertices) - 2 for p in hi.data.polygons), "hi_verts": len(hi.data.vertices)}
    if NOBAKE:
        return out
    maps = {}
    for kind, nm, samples, nc in (("NORMAL", "normal", 1, True), ("DIFFUSE", "albedo", 1, False), ("AO", "ao", ao_samples, True)):
        img = prec.bake_pass(low, hi, kind, tex, f"{tag}_{nm}", samples=samples, extrusion=0.012, max_ray=0.03, margin=8, non_color=nc)
        img.filepath_raw = rb.out(f"maps/{tag}_{nm}.png")
        img.file_format = "PNG"
        img.save()
        img.pack()
        maps[nm] = img
    out["maps"] = maps
    out["map_stats"] = {k: prec.image_stats(v) for k, v in maps.items()}
    low.data.materials.clear()
    low.data.materials.append(prec.pbr_material(tag + "_Skin", maps["albedo"], maps["normal"], maps["ao"], roughness=0.5))
    out["uv"] = uv_stats(low, tex)
    return out


head_skin = skin_pipeline(head, "Head", VOX, TEX, 11, 24)
lap("head_skin")
jaw_skin = skin_pipeline(jaw, "Jaw", JAW_VOX, JAW_TEX, 23, 16, size_far=0.0120)
lap("jaw_skin")
if abs(S["open_deg"]) > 1e-9:
    for ob in jaw_group + [jaw_skin["hi"]]:
        ob.data.transform(Mopen)
        ob.data.update()
for hi in (head_skin["hi"], jaw_skin["hi"]):
    hi.hide_render = True

# =============================================================================================== VERIFY
parts = [head, jaw, horn, teeth_up, teeth_lo, tongue] + eyes
for o in parts:
    o.data.update()

for o in [head, jaw, horn] + eyes:
    t = prec.topology(o)
    rep.count("topology", f"{o.name}: non-manifold edges", t["non_manifold_edges"], 0)
    rep.count("topology", f"{o.name}: separate islands", t["islands"], 2 if o is horn else 1)
for o, n in ((teeth_up, 2 * S["teeth_up"]), (teeth_lo, 2 * S["teeth_lo"])):
    t = prec.topology(o)
    rep.count("topology", f"{o.name}: closed spikes (islands)", t["islands"], n)
    rep.count("topology", f"{o.name}: non-manifold edges", t["non_manifold_edges"], 0)

last = LOG["sculpt_loop"][-1]
rep.num("sculpt loop", "eye protrusion beyond surface", last["eye_protrusion_mm"] / 1000, S["eye_protrusion"], 0.2)
rep.at_most("sculpt loop", "Kelvinlet constraint solve residual at the handles (head)", head_info["solve_residual_m"], 1e-3)
rep.at_most("sculpt loop", "Kelvinlet constraint solve residual at the handles (jaw)", jaw_info["solve_residual_m"], 1e-3)
rep.at_most("sculpt loop", "worst edit miss, head (requested vs measured)", head_info["max_err_mm"] / 1000, 0.15)
rep.at_most("sculpt loop", "worst edit miss, jaw (requested vs measured)", jaw_info["max_err_mm"] / 1000, 0.15)
disp = np.linalg.norm(HEAD_SCULPTED - HEAD_P0, axis=1)
rep.check("sculpt loop", "edits are local: untouched vertices stay put", np.percentile(disp, 25) < 0.0005,
          f"p25 move {np.percentile(disp, 25) * 1000:.3f} mm, max {disp.max() * 1000:.1f} mm")

for o in (head, jaw, horn):
    mean, mx = prec.symmetry(o)
    rep.at_most("symmetry (Mirror modifier)", f"{o.name}: mirror error max", mx, 0.001, unit="mm")

# -- the 1:1 check against the reference: absolute mapping, no registration
sub = [head, jaw, horn, teeth_up, teeth_lo]
m_final = prec.silhouette(sub, (CY, CZ), PPM, (RW, RH), SIL_TMP)
prec.overlay_png(m_final, REF_MASK, rb.out("dragon_ref_overlay.png"))
iou_full = prec.iou(m_final, REF_MASK)
top_f, bot_f = prec.outline(m_final)
both = ~np.isnan(top_f) & ~np.isnan(REF_TOP)
et_all = np.abs(top_f - REF_TOP)[both]
eb_all = np.abs(bot_f - REF_BOT)[both]
LOG["reference"] = {"iou": iou_full, "top_err_mean_px": float(et_all.mean()), "top_err_p95_px": float(np.percentile(et_all, 95)),
                    "bottom_err_mean_px": float(eb_all.mean()), "bottom_err_p95_px": float(np.percentile(eb_all, 95)),
                    "ref_area_px": int(REF_MASK.sum()), "model_area_px": int(m_final.sum()),
                    "ref_only_px": int((REF_MASK & ~m_final).sum()), "model_only_px": int((m_final & ~REF_MASK).sum())}
rep.at_least("reference silhouette", "IoU vs the reference side view (absolute mapping, no fitting)", iou_full, 0.93, unit="")
rep.at_most("reference silhouette", "top outline error, mean", float(et_all.mean()) / PPM, 4.0)
rep.at_most("reference silhouette", "top outline error, p95", float(np.percentile(et_all, 95)) / PPM, 12.0)
rep.at_most("reference silhouette", "bottom outline error, mean", float(eb_all.mean()) / PPM, 4.0)
rep.at_most("reference silhouette", "bottom outline error, p95", float(np.percentile(eb_all, 95)) / PPM, 12.0)
x_tip = int(np.nanargmax(np.where(np.isnan(top_f), -1, np.arange(RW))))
rep.num("reference silhouette", "horn tip x vs reference (px -> mm)", float(x_tip - LM["horn_tip"][0]) / PPM, 0.0, 6.0)
_bx = LM["brow_top"][0]
rep.num("reference silhouette", "brow top height above the orbit centre, model vs reference (silhouette)",
        float(LM["orbit_center"][1] - top_f[_bx]) / PPM, float(LM["orbit_center"][1] - REF_TOP[_bx]) / PPM, 3.0)

for name, placed, canine, n_each, tree in (("upper", up_placed, S["canine_up"], S["teeth_up"], prec.bvh_of(head)),
                                           ("lower", lo_placed, S["canine_lo"], S["teeth_lo"], JAW_SURF_BEFORE)):
    rep.count("teeth", f"{name}: tooth count", len(placed), 2 * n_each)
    rep.num("teeth", f"{name}: longest tooth = canine", max(l for _p, l in placed), canine, 0.01)
    worst = max(tree.find_nearest(V(p))[3] for p, _l in placed)
    rep.at_most("teeth", f"{name}: roots on surface (worst)", worst, 0.3)

allp = np.vstack([prec.coords(o) for o in parts])
bb = allp.max(0) - allp.min(0)
rep.num("dimensions", "snout tip y (the nose-bulb edit moves the tip by design)", float(prec.coords(head)[:, 1].min()), S["snout_tip_y"], 0.5)
_a = JAW_TIP_CLOSED - np.array(S["hinge"])
_b = prec.coords(jaw)[JAW_TIP_IDX] - np.array(S["hinge"])
rep.num("dimensions", "jaw open angle (measured from the jaw tip about the hinge)",
        math.degrees(math.atan2(_a[1] * _b[2] - _a[2] * _b[1], _a[1] * _b[1] + _a[2] * _b[2])), S["open_deg"], 0.01, unit="deg", scale=1.0)
rep.num("dimensions", "eye spacing (centre to centre, from the eye meshes)",
        float(prec.coords(eyes[0]).mean(0)[0] - prec.coords(eyes[1]).mean(0)[0]), 2 * S["eye_c"][0], 0.05)
rep.check("dimensions", "assembly bbox (m)", True, f"{bb[0]:.3f} x {bb[1]:.3f} x {bb[2]:.3f}")

hi_tree = prec.bvh_of(head_skin["hi"])
d = prec.distance_to(prec.coords(head), hi_tree)
rep.at_most("bake", "low-poly vs high-poly surface distance mean", float(d.mean()), 0.8)
rep.at_most("bake", "low-poly vs high-poly surface distance p99", float(np.percentile(d, 99)), 2.0)
rep.check("bake", "poly budget", True, f"low {prec.topology(head)['tris'] + prec.topology(jaw)['tris']} tris, high {head_skin['hi_tris'] + jaw_skin['hi_tris']} tris")
if not NOBAKE:
    ns = head_skin["map_stats"]["normal"]["mean"]
    rep.check("bake", "normal map is tangent-space (blue dominant)", ns[2] > 0.8 and 0.4 < ns[0] < 0.6, f"mean rgb {ns}")
    rep.check("bake", "ao map has occlusion range", head_skin["map_stats"]["ao"]["std"][0] > 0.02, f"std {head_skin['map_stats']['ao']['std'][0]}")
    uvs = head_skin["uv"]
    rep.check("bake", "texel density", uvs["px_per_mm_p5"] > 0.5, f"{uvs['px_per_mm_mean']:.2f} px/mm mean, p5 {uvs['px_per_mm_p5']:.2f}, p95 {uvs['px_per_mm_p95']:.2f}")
lap("verify")
print(rep.table())


# =============================================================================================== RENDER
def shot(name, cam, res, samples=48, denoise=True):
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = denoise
    sc.cycles.max_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = rb.out(name)
    bpy.ops.render.render(write_still=True)
    return sc.render.filepath


if not NORENDER:
    sc = bpy.context.scene
    focus = (0.0, -0.05, 0.09)
    R = 0.5 if QUICK else 1.0
    cams = dict(
        hero=prec.camera((1.15, -1.55, 0.55), focus, 75, "Hero"),
        front=prec.camera((0.0, -1.5, 0.09), focus, 85, "Front"),
        side=prec.camera((1.6, CY, CZ), (0, CY, CZ), 70, "Side"),
        top=prec.camera((0.0, -0.05, 1.5), (0, -0.05, 0.0), 80, "Top"),
        back=prec.camera((-0.95, 1.05, 0.50), (0, 0.0, 0.10), 75, "Back"),
        eye=prec.camera((0.55, -0.62, 0.30), (0.09, S["eye_c"][1], S["eye_c"][2]), 90, "EyeCam"),
        mouth=prec.camera((0.55, -0.75, -0.12), (0.0, -0.30, -0.01), 85, "MouthCam"),
        horn=prec.camera((0.55, 0.65, 0.45), (0.05, 0.15, 0.28), 85, "HornCam"),
    )
    shot("dragon_hero.png", cams["hero"], (int(1200 * R), int(1200 * R)), 24 if QUICK else 96)
    lap("hero")
    files = [shot(f"_v_{k}.png", cams[k], (int(520 * R), int(520 * R)), 16 if QUICK else 32) for k in ("front", "side", "top", "back")]
    prec.contact_sheet(files, rb.out("dragon_views.png"))
    files = [shot(f"_d_{k}.png", cams[k], (int(640 * R), int(640 * R)), 16 if QUICK else 40) for k in ("eye", "mouth", "horn")]
    prec.contact_sheet(files, rb.out("dragon_details.png"))
    lap("views")
    keep_vis = [(o, o.hide_render) for o in bpy.data.objects if o.type == "MESH" and o.name != "StudioFloor"]
    for o, _h in keep_vis:
        o.hide_render = True
    blockout.hide_render = False
    p1 = shot("_s_block.png", cams["hero"], (int(640 * R), int(640 * R)), 16 if QUICK else 40)
    blockout.hide_render = True
    head.hide_render = False
    saved_mat = list(head.data.materials)
    head.data.materials.clear()
    head.data.materials.append(clay)
    p2 = shot("_s_sculpt.png", cams["hero"], (int(640 * R), int(640 * R)), 16 if QUICK else 40)
    head.hide_render = True
    heat.hide_render = False
    p3 = shot("_s_heat.png", cams["hero"], (int(640 * R), int(640 * R)), 16 if QUICK else 40)
    heat.hide_render = True
    head.data.materials.clear()
    for m in saved_mat:
        head.data.materials.append(m)
    prec.contact_sheet([p1, p2, p3], rb.out("dragon_sculpt_steps.png"))
    for o, _h in keep_vis:
        o.hide_render = True
    head.hide_render = False
    p_low = shot("_c_low.png", cams["hero"], (int(700 * R), int(700 * R)), 16 if QUICK else 48)
    head.hide_render = True
    head_skin["hi"].hide_render = False
    p_hi = shot("_c_hi.png", cams["hero"], (int(700 * R), int(700 * R)), 16 if QUICK else 48)
    head_skin["hi"].hide_render = True
    prec.contact_sheet([p_hi, p_low], rb.out("dragon_hi_vs_low.png"))
    for o, h in keep_vis:
        o.hide_render = h
    for o in (blockout, heat, head_skin["hi"], jaw_skin["hi"]):
        o.hide_render = True
    prec.wireframe_render(rb.out("dragon_wireframe.png"), (int(900 * R), int(900 * R)), 8,
                          hide=[blockout, heat, head_skin["hi"], jaw_skin["hi"], bpy.data.objects["StudioFloor"]])
    sc.camera = cams["hero"]
    lap("render")

if not NOBAKE:
    prec.contact_sheet([rb.out(f"maps/Head_{k}.png") for k in ("albedo", "normal", "ao")], rb.out("dragon_maps.png"), max_h=512)

# =============================================================================================== EXPORT
for o in bpy.context.selected_objects:
    o.select_set(False)
for o in parts:
    o.select_set(True)
bpy.context.view_layer.objects.active = head
rb.export("dragon_head.glb", use_selection=True)
for o in bpy.context.selected_objects:
    o.select_set(False)
bpy.ops.wm.save_as_mainfile(filepath=rb.out("dragon_head.blend"), compress=True)
lap("export")

spec = {k: (list(v) if isinstance(v, tuple) else v) for k, v in S.items()}
spec["reference"] = REF["source"]
spec["mapping"] = {"ppm": PPM, "nose_tip_world_yz": [Y_N, Z_N], "landmarks_px": LM, "scheme_px": REF["scheme_px"]}
spec["head_edits"] = head_edits(params)
spec["jaw_edits"] = JAW_EDITS
with open(rb.out("dragon_spec.json"), "w") as fh:
    json.dump(spec, fh, indent=2, default=float)
with open(rb.out("dragon_report.json"), "w") as fh:
    json.dump({"rows": rep.rows, "ok": rep.ok(), "loops": LOG, "timing": timing,
               "head": {"hi_tris": head_skin["hi_tris"], "uv": head_skin.get("uv"), "maps": head_skin.get("map_stats")},
               "jaw": {"hi_tris": jaw_skin["hi_tris"]}}, fh, indent=2, default=float)
with open(rb.out("dragon_report.txt"), "w") as fh:
    fh.write(rep.table() + "\n")
rb.report(dragon={"checks": len(rep.rows), "pass": int(sum(r["ok"] for r in rep.rows)), "ok": bool(rep.ok()), "iou": iou_full, "seconds": timing})
if not rep.ok():
    raise RuntimeError("dragon spec checks failed: " + "; ".join(r["name"] for r in rep.rows if not r["ok"]))
