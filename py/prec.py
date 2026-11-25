"""prec: precision toolkit for headless Blender (numpy + bmesh + BVH). Works without any window.

Sculpting     Regularized Kelvinlets (de Goes & James, Pixar Technical Memo #17-03, 2017): closed-form
              elastic brushes, multi-scale falloff, and an exact pointwise-constraint solve, so
              "move this surface point by exactly 8 mm, keep those points fixed" is one linear solve.
Measuring     ray casts, topology, symmetry, distance to a reference, a pass/fail Report.
Placing       equal arc-length placement and surface-snapped tubes/cones (horns, teeth, spines).
Texturing     Worley (cellular) scale patterns, high->low baking, PBR material with glTF-safe maps.
"""
import json
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

V = Vector


# ----------------------------------------------------------------------------- mesh <-> numpy
def coords(ob):
    n = len(ob.data.vertices)
    a = np.empty(n * 3, np.float32)
    ob.data.vertices.foreach_get("co", a)
    return a.reshape(n, 3).astype(np.float64)


def set_coords(ob, P):
    ob.data.vertices.foreach_set("co", np.asarray(P, np.float32).ravel())
    ob.data.update()


def vertex_normals(ob):
    n = len(ob.data.vertices)
    a = np.empty(n * 3, np.float32)
    ob.data.vertex_normals.foreach_get("vector", a)
    return a.reshape(n, 3).astype(np.float64)


def new_object(name, bm, mats=(), smooth=True, collection=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    return ob


def duplicate(ob, name):
    c = ob.copy()
    c.data = ob.data.copy()
    c.name = name
    bpy.context.scene.collection.objects.link(c)
    return c


# ----------------------------------------------------------------------------- Regularized Kelvinlets
def _ab(nu, mu=1.0):
    a = 1.0 / (4.0 * math.pi * mu)
    return a, a / (4.0 * (1.0 - nu))


def _scales(eps, order):
    """Radial scales and weights of the (multi-scale) brush: order 1 plain, 2 bi-scale, 3 tri-scale
    (eps_{i+1} = 1.1 eps_i, as in the paper). Higher order = faster, more local falloff."""
    if order <= 1:
        return [eps], [1.0]
    if order == 2:
        return [eps, 1.1 * eps], [1.0, -1.0]
    e1, e2, e3 = eps, 1.1 * eps, 1.21 * eps
    w3 = (e2 * e2 - e1 * e1) / (e3 * e3 - e2 * e2)
    return [e1, e2, e3], [1.0, -1.0 - w3, w3]


def kernel(r, eps, nu=0.45, order=3):
    """K(r) of Eq. 6 summed over the multi-scale combination. r (...,3) -> (...,3,3)."""
    a, b = _ab(nu)
    r2 = (r * r).sum(-1)
    rr = r[..., :, None] * r[..., None, :]
    I = np.eye(3)
    K = np.zeros(r.shape[:-1] + (3, 3))
    es, ws = _scales(eps, order)
    for e, w in zip(es, ws):
        re = np.sqrt(r2 + e * e)
        K += w * (((a - b) / re + a * e * e / (2 * re ** 3))[..., None, None] * I + (b / re ** 3)[..., None, None] * rr)
    return K


def tip_gain(eps, nu=0.45, order=3):
    """Scalar k0 with K(0) = k0 I (the displacement at the brush centre per unit force)."""
    a, b = _ab(nu)
    es, ws = _scales(eps, order)
    return sum(w * (3 * a - 2 * b) / (2 * e) for e, w in zip(es, ws))


def field(P, centers, forces, eps, nu=0.45, order=3):
    """Displacement of points P due to point forces `forces` at `centers` (Eq. 6, superposed)."""
    centers = np.atleast_2d(centers)
    forces = np.atleast_2d(forces)
    eps = np.broadcast_to(np.asarray(eps, float), (len(centers),))
    a, b = _ab(nu)
    U = np.zeros_like(P)
    for c, f, e0 in zip(centers, forces, eps):
        r = P - c
        r2 = (r * r).sum(1)
        rf = r @ f
        es, ws = _scales(e0, order)
        for e, w in zip(es, ws):
            re = np.sqrt(r2 + e * e)
            U += w * (((a - b) / re + a * e * e / (2 * re ** 3))[:, None] * f[None, :] + (b / re ** 3 * rf)[:, None] * r)
    return U


def grab(P, x0, u, eps, nu=0.45, order=3):
    """Elastic grab brush: the point x0 moves by exactly u, everything else follows elastically."""
    f = np.asarray(u, float) / tip_gain(eps, nu, order)
    return P + field(P, [x0], [f], [eps], nu, order)


def solve_forces(H, D, eps, nu=0.45, order=3, reg=0.0):
    """Forces such that the displacement field equals D[i] at every handle H[i] (single linear solve,
    Sec. 5/7 of the paper). Handles with D=0 are pins that hold the surface still. `reg` is a Tikhonov term
    relative to the mean diagonal; it keeps the forces bounded when handles are close to each other."""
    H = np.asarray(H, float)
    D = np.asarray(D, float)
    M = len(H)
    eps = np.broadcast_to(np.asarray(eps, float), (M,))
    A = np.zeros((3 * M, 3 * M))
    for i in range(M):
        Ki = kernel(H - H[i], eps[i], nu, order)  # block (j, i)
        A[:, 3 * i:3 * i + 3] = Ki.transpose(0, 1, 2).reshape(3 * M, 3)
    if reg:
        A = A + reg * np.mean(np.diag(A)) * np.eye(3 * M)
    try:
        f = np.linalg.solve(A, D.ravel())
    except np.linalg.LinAlgError:
        f = np.linalg.lstsq(A, D.ravel(), rcond=None)[0]
    F = f.reshape(M, 3)
    res = np.abs(field(H, H, F, eps, nu, order) - D).max()
    return F, float(res)


def solve_stable(H, D, eps, P, max_disp, nu=0.45, order=3, regs=(0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3)):
    """solve_forces with the least regularisation whose displacement field on the mesh points P stays finite and
    below max_disp. Returns (forces, handle residual, reg used, displacement of P)."""
    for reg in regs:
        F, res = solve_forces(H, D, eps, nu, order, reg)
        U = field(P, H, F, eps, nu, order)
        if np.isfinite(U).all() and np.abs(U).max() <= max_disp:
            break
    return F, res, reg, U


def mirror_map(P, axis=0):
    """Index of the mirrored vertex for every vertex (nearest vertex of the reflected position)."""
    from mathutils.kdtree import KDTree
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(p, i)
    kd.balance()
    m = np.ones(3)
    m[axis] = -1.0
    return np.array([kd.find(V(p))[1] for p in P * m])


def symmetrize(U, idx, axis=0):
    """Average a displacement field with its mirror image: exact vertex symmetry. Because the constraints are
    linear and mirror-symmetric, the average still satisfies every one of them."""
    m = np.ones(3)
    m[axis] = -1.0
    return 0.5 * (U + U[idx] * m)


def pin_points(P, avoid, avoid_r, spacing, seed=3, mirror_axis=0):
    """Mirror-symmetric Poisson-disk sample of mesh vertices to use as zero-displacement pins, never
    closer than avoid_r[i] to avoid[i]."""
    rng = np.random.default_rng(seed)
    avoid = np.asarray(avoid, float).reshape(-1, 3)
    avoid_r = np.asarray(avoid_r, float)
    ok = (np.abs(P[:, mirror_axis]) < 1e-6) | (P[:, mirror_axis] > spacing / 2)
    for p, r in zip(avoid, avoid_r):
        ok &= np.linalg.norm(P - p, axis=1) > r
    idx = rng.permutation(np.nonzero(ok)[0])
    acc = []
    for i in idx:
        if not acc or np.linalg.norm(np.array([P[j] for j in acc]) - P[i], axis=1).min() > spacing:
            acc.append(i)
    pins = [P[i] for i in acc]
    for i in acc:
        if P[i][mirror_axis] > 1e-6:
            m = P[i].copy()
            m[mirror_axis] *= -1
            pins.append(m)
    return np.array(pins)


# ----------------------------------------------------------------------------- measuring
def bvh_of(ob):
    return BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())


def ray(tree, origin, direction, maxd=100.0):
    loc, nor, _i, dist = tree.ray_cast(V(origin), V(direction).normalized(), maxd)
    return (loc, nor, dist) if loc is not None else (None, None, None)


def topology(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bad = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    seen, islands = set(), 0
    for v in bm.verts:
        if v.index in seen:
            continue
        islands += 1
        stack = [v]
        seen.add(v.index)
        while stack:
            cur = stack.pop()
            for e in cur.link_edges:
                u = e.other_vert(cur)
                if u.index not in seen:
                    seen.add(u.index)
                    stack.append(u)
    vol = bm.calc_volume(signed=True)
    out = {"verts": len(bm.verts), "tris": sum(len(f.verts) - 2 for f in bm.faces), "non_manifold_edges": bad,
           "islands": islands, "volume_m3": vol}
    bm.free()
    return out


def symmetry(ob, axis=0):
    """Mesh symmetry: distance (m) from every mirrored vertex to its nearest vertex. Returns (mean, max).
    Vertex-based on purpose: a triangulated quad mesh is never exactly mirror-symmetric as a *surface*."""
    from mathutils.kdtree import KDTree
    P = coords(ob)
    kd = KDTree(len(P))
    for i, p in enumerate(P):
        kd.insert(p, i)
    kd.balance()
    m = np.array([1.0, 1.0, 1.0])
    m[axis] = -1.0
    d = np.array([kd.find(V(p))[2] for p in P * m])
    return float(d.mean()), float(d.max())


def distance_to(points, tree):
    d = np.empty(len(points))
    for i, p in enumerate(points):
        loc, _n, _i, dist = tree.find_nearest(V(p))
        d[i] = dist
    return d


class Report:
    """Pass/fail table of measured vs specified values (units converted for display)."""

    def __init__(self):
        self.rows = []

    def num(self, group, name, measured, target, tol, unit="mm", scale=1000.0):
        err = float((measured - target) * scale)
        self.rows.append({"group": group, "name": name, "measured": float(measured * scale), "target": float(target * scale),
                          "tol": tol, "unit": unit, "err": err, "ok": bool(abs(err) <= tol)})

    def count(self, group, name, measured, target):
        self.rows.append({"group": group, "name": name, "measured": int(measured), "target": int(target), "tol": 0,
                          "unit": "", "err": int(measured - target), "ok": bool(measured == target)})

    def check(self, group, name, cond, detail=""):
        self.rows.append({"group": group, "name": name, "measured": str(detail), "target": "", "tol": "", "unit": "",
                          "err": "", "ok": bool(cond)})

    def at_most(self, group, name, measured, limit, unit="mm", scale=1000.0):
        self.rows.append({"group": group, "name": name, "measured": float(measured * scale), "target": f"<= {limit}",
                          "tol": "", "unit": unit, "err": "", "ok": bool(measured * scale <= limit)})

    def at_least(self, group, name, measured, limit, unit=""):
        self.rows.append({"group": group, "name": name, "measured": float(measured), "target": f">= {limit}",
                          "tol": "", "unit": unit, "err": "", "ok": bool(measured >= limit)})

    def ok(self):
        return all(r["ok"] for r in self.rows)

    def table(self):
        lines, g = [], None
        for r in self.rows:
            if r["group"] != g:
                g = r["group"]
                lines.append(f"-- {g}")

            def f(x):
                return f"{x:.3f}" if isinstance(x, float) else str(x)
            lines.append(f"  [{' ok ' if r['ok'] else 'FAIL'}] {r['name']:<46} measured {f(r['measured']):>10} {r['unit']:<2}"
                         f"  target {f(r['target']):>10}  err {f(r['err']):>8}  tol {r['tol']}")
        lines.append(f"{sum(r['ok'] for r in self.rows)}/{len(self.rows)} checks pass")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- placing / building
def hermite1d(xs, ys, x):
    """C1 cubic Hermite interpolation with finite-difference tangents (smooth stations -> curve)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m = np.gradient(ys, xs)
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h
    t2, t3 = t * t, t ** 3
    return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1]
            + (t3 - t2) * h * m[i + 1])


def polyline_length(pts):
    pts = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def arc_positions(pts, n, margin=0.0):
    """n points at equal arc-length spacing along polyline pts (margin = fraction trimmed at both ends)."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(margin, 1 - margin, n) * s[-1]
    return np.stack([np.interp(t, s, pts[:, k]) for k in range(3)], axis=1), t / s[-1]


def bezier(p0, p1, p2, p3, n=64):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def tube_bm(bm, path, radii, n=12, ridge=None, cap_start=True, cap_end=True, squash=1.0, uv=False, axis=None):
    """Tapered tube along a polyline (parallel-transport frames). radii: per point. ridge(i, theta) multiplies.
    axis: keep every ring in a plane perpendicular to this direction instead of to the path tangent. Use it when the
    outline along that axis must equal the radii exactly (a fast taper over a sloping path otherwise tilts the rings
    and bloats the silhouette by about r*sin(slope))."""
    path = [V(p) for p in path]
    N = len(path)
    radii = list(radii)
    if axis is not None:
        t = [V(axis).normalized() for _ in range(N)]
    else:
        t = [(path[min(i + 1, N - 1)] - path[max(i - 1, 0)]).normalized() for i in range(N)]
    ref = V((0, 0, 1)) if abs(t[0].z) < 0.9 else V((1, 0, 0))
    nrm = (ref - t[0] * ref.dot(t[0])).normalized()
    uvl = bm.loops.layers.uv.verify() if uv else None
    rings = []
    for i in range(N):
        nrm = (nrm - t[i] * nrm.dot(t[i]))
        nrm = nrm.normalized() if nrm.length > 1e-9 else V((1, 0, 0))
        bn = t[i].cross(nrm)
        ring = []
        for j in range(n):
            th = math.tau * j / n
            k = ridge(i, th) if ridge else 1.0
            ring.append(bm.verts.new(path[i] + nrm * (radii[i] * k * math.cos(th)) + bn * (radii[i] * k * squash * math.sin(th))))
        rings.append(ring)
    made = []
    for i in range(N - 1):
        for j in range(n):
            j2 = (j + 1) % n
            f = bm.faces.new((rings[i][j], rings[i][j2], rings[i + 1][j2], rings[i + 1][j]))
            made.append(f)
            if uv:
                for loop, (ii, jj) in zip(f.loops, ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))):
                    loop[uvl].uv = (jj / n, ii / max(N - 1, 1))
    if cap_start:
        made.append(bm.faces.new(rings[0][::-1]))
    if cap_end:
        made.append(bm.faces.new(rings[-1]))
    c = sum((v.co for r in rings for v in r), V()) / (N * n)
    if sum(f.normal.dot(f.calc_center_median() - c) * f.calc_area() for f in made) < 0:
        bmesh.ops.reverse_faces(bm, faces=made)
    return rings


def tube(name, path, radii, mats=(), n=12, **kw):
    bm = bmesh.new()
    tube_bm(bm, path, radii, n=n, **kw)
    return new_object(name, bm, mats)


# ----------------------------------------------------------------------------- lofts, targets, fitting
def station_vals(ST, y):
    return [hermite1d(ST[:, 0], ST[:, k], y) for k in range(1, 5)]


def loft_half(name, ST, n_rings, n_around, mat):
    """The x>=0 half of a superelliptic loft (clean quads, grid UVs) plus a Mirror modifier: symmetric by
    construction. ST rows are (y, half-width, z top, z bottom, exponent). UV u runs 0..0.5 on this half; the
    modifier mirrors U so the full body spans 0..1."""
    ys = np.linspace(ST[0, 0], ST[-1, 0], n_rings)
    half = n_around // 2
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    rows = []
    for y in ys:
        w, zt, zb, ne = station_vals(ST, y)
        zc, hz = (zt + zb) / 2, (zt - zb) / 2
        row = []
        for j in range(half + 1):
            phi = math.tau * j / n_around
            s, c = math.sin(phi), math.cos(phi)
            x = 0.0 if j in (0, half) else w * abs(s) ** (2 / ne)
            g = -np.sign(c) * abs(c) ** (2 / ne)
            row.append(bm.verts.new((x, float(y), float(zc + g * hz))))
        rows.append(row)
    for i in range(n_rings - 1):
        for j in range(half):
            f = bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
            for loop, (ii, jj) in zip(f.loops, ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))):
                loop[uvl].uv = (jj / n_around, ii / (n_rings - 1))
    for ring in (rows[0], rows[-1]):
        f = bm.faces.new(ring)
        for loop in f.loops:
            loop[uvl].uv = (0.25, 0.5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    c = sum((v.co for v in bm.verts), V()) / len(bm.verts)
    c.x = 0.0
    if sum(f.normal.dot(f.calc_center_median() - c) * f.calc_area() for f in bm.faces) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    ob = new_object(name, bm, [mat])
    add_mirror(ob)
    return ob


def target_mesh(name, P, T):
    """Plain triangle mesh from arrays: a measurement target, never part of the model."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in P], [], [tuple(int(i) for i in t) for t in T])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def sample_surface(ob, n, seed=0):
    """n area-weighted random points on the evaluated surface of ob (modifiers applied, world space)."""
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    me.calc_loop_triangles()
    Vs = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", Vs)
    Vs = Vs.reshape(-1, 3)
    tri = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    ev.to_mesh_clear()
    M = np.array(ob.matrix_world)
    Vs = Vs @ M[:3, :3].T + M[:3, 3]
    a, b, c = Vs[tri[:, 0]], Vs[tri[:, 1]], Vs[tri[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return a[idx] * (1 - r1)[:, None] + b[idx] * (r1 * (1 - r2))[:, None] + c[idx] * (r1 * r2)[:, None]


def nearest_points(P, tree):
    """Closest surface point and distance for every row of P."""
    Q = np.empty((len(P), 3))
    d = np.empty(len(P))
    for i, p in enumerate(P):
        loc, _n, _i, dist = tree.find_nearest(V(p))
        Q[i], d[i] = loc, dist
    return Q, d


def chamfer(model, target, n=5000, seed=0):
    """Two-sided surface distance between model and target (both objects). accuracy = model->target (is what I built
    on the target?), completeness = target->model (is the target covered?). Distances in metres."""
    d_mt = nearest_points(sample_surface(model, n, seed), bvh_of(target))[1]
    d_tm = nearest_points(sample_surface(target, n, seed + 1), bvh_of(model))[1]
    f = lambda d: dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)), max=float(d.max()))
    return dict(accuracy=f(d_mt), completeness=f(d_tm), chamfer_mean=float((d_mt.mean() + d_tm.mean()) / 2))


def roughness(ob):
    """Mean dihedral angle (degrees) over the interior edges of the object's own mesh (modifiers ignored). A fit that
    matches a surface by crumpling it scores high here even when the distance metrics look good."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    ang = [e.calc_face_angle(0.0) for e in bm.edges if len(e.link_faces) == 2]
    bm.free()
    return math.degrees(float(np.mean(ang))) if ang else 0.0


def fit_to_target(ob, target_tree, spacing, iters=3, eps_k=1.6, gain=0.9, max_step=0.02, nu=0.45, direction="closest", reach=0.05,
                  robust_k=8, reg=1e-2, max_roughness=None):
    """Pull the half mesh of `ob` (x>=0, Mirror modifier) onto a target surface with Regularized Kelvinlets: one
    handle per `spacing` cell, mirrored handles for the other side, one stable constraint solve per iteration.
    direction='closest': straight to the closest target point. 'normal': along the vertex normal to the nearest ray hit
    within `reach` (either way); handles with no hit are skipped.
    robust_k: each handle takes the component-wise MEDIAN pull of its robust_k nearest handles. Neighbouring handles
    can hit different sheets (outer skin, inner mouth, a spike) and an interpolating solve would follow that noise.
    max_roughness (degrees, mean dihedral angle): an iterate rougher than this is never kept, so accuracy is pursued
    inside a smoothness budget. Keeps the best admissible iterate by mean vertex distance. Returns a log."""
    from mathutils.kdtree import KDTree
    M = np.array([-1.0, 1.0, 1.0])
    log = []
    best = (float("inf"), None)
    for it in range(iters + 1):
        P = coords(ob)
        _Q, dall = nearest_points(P, target_tree)
        cur = float(dall.mean())
        rough = roughness(ob)
        log.append(dict(iter=it, mean_mm=cur * 1000, p95_mm=float(np.percentile(dall, 95)) * 1000, max_mm=float(dall.max()) * 1000,
                        roughness_deg=rough))
        if cur < best[0] and (max_roughness is None or rough <= max_roughness or best[1] is None):
            best = (cur, P.copy())
        if it == iters:
            break
        cell = np.floor(P / spacing).astype(int)
        _u, first = np.unique(cell, axis=0, return_index=True)
        Hh = P[first]
        if direction == "normal":
            nrm = np.empty(len(P) * 3)
            ob.data.vertex_normals.foreach_get("vector", nrm)
            Nh = nrm.reshape(-1, 3)[first]
            D = np.zeros_like(Hh)
            hit = np.zeros(len(Hh), bool)
            for i, (h, n) in enumerate(zip(Hh, Nh)):
                pick = None
                for s in (1.0, -1.0):
                    loc, _nn, _fi, dist = target_tree.ray_cast(V(h) + V(n * s) * 1e-4, V(n * s), reach)
                    if loc is not None and (pick is None or dist < pick[0]):
                        pick = (dist, s)
                if pick:
                    D[i] = n * pick[1] * pick[0] * gain
                    hit[i] = True
            Hh, D = Hh[hit], D[hit]
        else:
            Q, _d = nearest_points(Hh, target_tree)
            D = Q - Hh
            L = np.linalg.norm(D, axis=1)
            D = D * np.minimum(1.0, max_step / np.maximum(L, 1e-12))[:, None] * gain
        if len(Hh) < 8:
            break
        if robust_k and len(Hh) > robust_k:
            kd = KDTree(len(Hh))
            for i, h in enumerate(Hh):
                kd.insert(V(h), i)
            kd.balance()
            D = np.array([np.median(D[[j for (_c, j, _d) in kd.find_n(V(h), robust_k)]], axis=0) for h in Hh])
        off = Hh[:, 0] > spacing * 0.25
        H = np.vstack([Hh, Hh[off] * M])
        Dm = np.vstack([D, D[off] * M])
        F, res, reg_used, U = solve_stable(H, Dm, np.full(len(H), eps_k * spacing), P, max_disp=3 * max_step, nu=nu,
                                           regs=tuple(reg * 10.0 ** k for k in range(6)) if reg else (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3))
        U[np.abs(P[:, 0]) < 1e-9, 0] = 0.0
        set_coords(ob, P + U)
        log[-1].update(handles=len(H), regularisation=reg_used, solve_residual_m=res)
    set_coords(ob, best[1])
    return log


def fit_coarse_to_fine(ob, target_tree, schedule, **kw):
    """fit_to_target over a schedule of (spacing, iterations): wide, smooth moves first, local detail last. The
    roughness budget (max_roughness) is passed through to every stage. Returns the concatenated log."""
    log = []
    for spacing, iters in schedule:
        stage = fit_to_target(ob, target_tree, spacing, iters=iters, **kw)
        for r in stage:
            r["spacing_mm"] = spacing * 1000
        log += stage
    return log


# ----------------------------------------------------------------------------- Worley scales
def _hash01(c, seed, k):
    x = (c[:, 0].astype(np.uint64) * np.uint64(73856093)) ^ (c[:, 1].astype(np.uint64) * np.uint64(19349663)) \
        ^ (c[:, 2].astype(np.uint64) * np.uint64(83492791)) ^ np.uint64(seed * 2654435761 + k * 40503)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def worley(P, size, seed=1, jitter=0.9):
    """Cellular noise. P (N,3), size scalar or (N,) -> F1, F2 (in cells) and an id per nearest cell (0..1)."""
    size = np.asarray(size, float)
    q = P / (size[:, None] if size.ndim else size)
    ci = np.floor(q).astype(np.int64)
    F1 = np.full(len(P), 9.0)
    F2 = np.full(len(P), 9.0)
    ID = np.zeros(len(P))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = ci + np.array([dx, dy, dz])
                pt = c + 0.5 + (np.stack([_hash01(c, seed, k) for k in range(3)], axis=1) - 0.5) * jitter
                d = np.linalg.norm(q - pt, axis=1)
                closer = d < F1
                F2 = np.minimum(F2, np.maximum(F1, d))
                ID = np.where(closer, _hash01(c, seed, 7), ID)
                F1 = np.minimum(F1, d)
    return F1, F2, ID


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- baking / materials
def new_image(name, size, color=(0, 0, 0, 1), non_color=False):
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.generated_color = color
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def bake_pass(low, high, kind, size, name, samples=1, extrusion=0.01, max_ray=0.03, margin=8, non_color=False):
    """Bake `kind` (NORMAL, AO, DIFFUSE colour) from `high` onto `low`'s UVs. low needs a material."""
    img = new_image(name, size, non_color=non_color)
    nt = low.data.materials[0].node_tree
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    nt.nodes.active = node
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    bk = sc.render.bake
    bk.use_selected_to_active = True
    bk.cage_extrusion = extrusion
    bk.max_ray_distance = max_ray
    bk.margin = margin
    bk.use_clear = True
    for o in bpy.context.selected_objects:
        o.select_set(False)
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    if kind == "DIFFUSE":
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    else:
        bpy.ops.object.bake(type=kind)
    nt.nodes.remove(node)
    return img


def image_stats(img):
    a = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(-1, 4)[:, :3]
    return {"mean": [round(float(x), 3) for x in a.mean(0)], "std": [round(float(x), 3) for x in a.std(0)]}


def pbr_material(name, albedo, normal=None, ao=None, roughness=0.5, metallic=0.0, spec=0.4):
    """Principled material from baked maps, with an AO texture wired to the 'glTF Material Output' group so the
    glTF exporter writes occlusionTexture."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = albedo
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if normal is not None:
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    if ao is not None:
        ta = nt.nodes.new("ShaderNodeTexImage")
        ta.image = ao
        grp = bpy.data.node_groups.get("glTF Material Output")
        if grp is None:
            grp = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
            grp.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        g = nt.nodes.new("ShaderNodeGroup")
        g.node_tree = grp
        g.location = (300, -400)
        nt.links.new(ta.outputs["Color"], g.inputs["Occlusion"])
    return m


def simple_material(name, color, rough=0.5, metal=0.0, spec=0.4, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color[:3], 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    if emission:
        b.inputs["Emission Color"].default_value = (*emission[:3], 1)
        b.inputs["Emission Strength"].default_value = 1.0
    return m


def image_from_array(name, arr, non_color=False, pack=True):
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), arr.dtype)], axis=2)
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"  # must precede the pixel write (it resets the buffer)
    img.pixels.foreach_set(np.clip(arr, 0, 1).astype(np.float32).ravel())
    if pack:
        img.pack()
    return img


# ----------------------------------------------------------------------------- studio
def studio(top=(0.16, 0.19, 0.27), bottom=(0.03, 0.035, 0.05), floor=True, floor_z=0.0, floor_r=3.0):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("Studio")
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out, bg = nt.nodes.new("ShaderNodeOutputWorld"), nt.nodes.new("ShaderNodeBackground")
    ramp, coord, sep = nt.nodes.new("ShaderNodeValToRGB"), nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ")
    mp = nt.nodes.new("ShaderNodeMapRange")
    mp.inputs["From Min"].default_value, mp.inputs["From Max"].default_value = -0.3, 0.9
    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mp.inputs["Value"])
    nt.links.new(mp.outputs["Result"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].color = (*bottom, 1)
    ramp.color_ramp.elements[1].color = (*top, 1)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    sc.world = w
    sc.view_settings.view_transform = "Standard"
    if floor:
        bpy.ops.mesh.primitive_circle_add(vertices=64, radius=floor_r, fill_type="NGON", location=(0, 0, floor_z))
        f = bpy.context.object
        f.name = "StudioFloor"
        f.data.materials.append(simple_material("StudioFloor", (0.10, 0.11, 0.13), rough=0.8, spec=0.2))


def area_light(name, loc, target, energy, size, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.size, d.color = energy, size, color
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (V(target) - V(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def camera(loc, target, lens=70, name="Cam", ortho=None):
    d = bpy.data.cameras.new(name)
    d.clip_start, d.clip_end = 0.01, 50
    if ortho:
        d.type, d.ortho_scale = "ORTHO", ortho
    else:
        d.lens = lens
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (V(target) - V(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def contact_sheet(paths, out_path, pad=6, bg=(0.03, 0.035, 0.05), max_h=None):
    """Tile PNGs left-to-right into one image; max_h block-averages tiles taller than this."""
    tiles = []
    for p in paths:
        im = bpy.data.images.load(p)
        w, h = im.size
        buf = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(buf)
        t = buf.reshape(h, w, 4)[..., :3]
        if max_h and h > max_h:
            k = int(math.ceil(h / max_h))
            hh, ww = (h // k) * k, (w // k) * k
            t = t[:hh, :ww].reshape(hh // k, k, ww // k, k, 3).mean(axis=(1, 3))
        tiles.append(t)
        bpy.data.images.remove(im)
    h = max(t.shape[0] for t in tiles)
    cols = []
    for t in tiles:
        c = np.ones((h, t.shape[1], 3), np.float32) * np.array(bg, np.float32)
        c[:t.shape[0]] = t
        cols += [c, np.ones((h, pad, 3), np.float32) * np.array(bg, np.float32)]
    sheet = np.concatenate(cols[:-1], axis=1)
    img = image_from_array("sheet", sheet, pack=False)
    img.filepath_raw = out_path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def wireframe_render(out_path, res, samples=8, hide=()):
    """Topology proof: Cycles with a wireframe-shader material override on the whole view layer."""
    sc = bpy.context.scene
    m = bpy.data.materials.new("WireOverride")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out, wire = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 0.9
    mix, emi = nt.nodes.new("ShaderNodeMix"), nt.nodes.new("ShaderNodeEmission")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.02, 0.03, 0.05, 1)
    mix.inputs["B"].default_value = (0.55, 0.85, 1.0, 1)
    nt.links.new(wire.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    vl = bpy.context.view_layer
    vl.material_override = m
    hidden = [(o, o.hide_render) for o in hide]
    for o in hide:
        o.hide_render = True
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    vl.material_override = None
    for o, h in hidden:
        o.hide_render = h


def heat_rgba(values, vmax):
    """Blue -> cyan -> green -> yellow -> red ramp for values in [0, vmax] (e.g. displacement in metres)."""
    t = np.clip(np.asarray(values, float) / vmax, 0, 1)
    stops = np.array([[0.05, 0.10, 0.55], [0.0, 0.65, 0.85], [0.10, 0.75, 0.25], [0.95, 0.85, 0.10], [0.90, 0.12, 0.08]])
    x = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(t, x, stops[:, k]) for k in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((len(t), 1))], axis=1)


def vertex_color_material(name, layer, rough=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    vc = m.node_tree.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = layer
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    m.node_tree.links.new(vc.outputs["Color"], b.inputs["Base Color"])
    return m


# ----------------------------------------------------------------------------- procedural skin and colour balance
def _node(nt, kind, **props):
    n = nt.nodes.new(kind)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _ramp(nt, stops, interp="LINEAR"):
    n = nt.nodes.new("ShaderNodeValToRGB")
    r = n.color_ramp
    r.interpolation = interp
    while len(r.elements) < len(stops):
        r.elements.new(0.5)
    for el, (p, c) in zip(r.elements, stops):
        el.position = p
        el.color = (*c[:3], 1.0)
    return n


def scale_skin_material(name, pal, cell=0.018, bump=0.6):
    """Procedural scaled skin (no UVs needed, works on tubes too). Voronoi cells are the scales; the body colour runs
    from back to belly with the world normal; every cell gets its own tint; grout is darker, rougher and may glow.
    pal keys: top, side, belly, grout (RGB), rough_plate, rough_grout, glow (emission strength in the grout, 0 = none),
    tint (per-cell colour variation 0..1), mottle (large-scale value variation 0..1)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial")
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled")
    tc = _node(nt, "ShaderNodeTexCoord")
    edge = _node(nt, "ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="DISTANCE_TO_EDGE")
    cellv = _node(nt, "ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1")
    for v in (edge, cellv):
        v.inputs["Scale"].default_value = 1.0 / cell
        v.inputs["Randomness"].default_value = 1.0
        nt.links.new(tc.outputs["Object"], v.inputs["Vector"])
    plate = _ramp(nt, [(pal.get("grout_w", 0.012), (0, 0, 0)), (pal.get("grout_w", 0.012) + 0.07, (1, 1, 1))])  # 0 grout, 1 plate
    nt.links.new(edge.outputs["Distance"], plate.inputs["Fac"])
    geo = _node(nt, "ShaderNodeNewGeometry")
    sep = _node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs["Vector"])
    up = _node(nt, "ShaderNodeMath", operation="MULTIPLY_ADD")
    nt.links.new(sep.outputs["Z"], up.inputs[0])
    up.inputs[1].default_value = 0.5
    up.inputs[2].default_value = 0.5
    body = _ramp(nt, [(0.0, pal["belly"]), (0.34, pal["belly"]), (0.52, pal["side"]), (0.80, pal["top"]), (1.0, pal["top"])])
    nt.links.new(up.outputs["Value"], body.inputs["Fac"])
    tint = _node(nt, "ShaderNodeMixRGB", blend_type="MULTIPLY")
    tint.inputs["Fac"].default_value = pal.get("tint", 0.35)
    nt.links.new(body.outputs["Color"], tint.inputs["Color1"])
    nt.links.new(cellv.outputs["Color"], tint.inputs["Color2"])
    noise = _node(nt, "ShaderNodeTexNoise", noise_dimensions="3D")
    noise.inputs["Scale"].default_value = 5.0
    noise.inputs["Detail"].default_value = 4.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    mot = _node(nt, "ShaderNodeMixRGB", blend_type="MULTIPLY")
    mot.inputs["Fac"].default_value = pal.get("mottle", 0.25)
    nt.links.new(tint.outputs["Color"], mot.inputs["Color1"])
    nt.links.new(noise.outputs["Color"], mot.inputs["Color2"])
    skin = _node(nt, "ShaderNodeMixRGB", blend_type="MIX")
    nt.links.new(plate.outputs["Color"], skin.inputs["Fac"])
    skin.inputs["Color1"].default_value = (*pal["grout"][:3], 1.0)
    nt.links.new(mot.outputs["Color"], skin.inputs["Color2"])
    nt.links.new(skin.outputs["Color"], bsdf.inputs["Base Color"])
    bw = _node(nt, "ShaderNodeRGBToBW")
    nt.links.new(plate.outputs["Color"], bw.inputs["Color"])
    rough = _node(nt, "ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = pal.get("rough_grout", 0.8)
    rough.inputs["To Max"].default_value = pal.get("rough_plate", 0.38)
    nt.links.new(bw.outputs["Val"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bp = _node(nt, "ShaderNodeBump")
    bp.inputs["Strength"].default_value = bump
    bp.inputs["Distance"].default_value = 0.004
    nt.links.new(bw.outputs["Val"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = pal.get("spec", 0.5)
    if pal.get("glow", 0.0) > 0:
        inv = _node(nt, "ShaderNodeMath", operation="SUBTRACT")
        inv.inputs[0].default_value = 1.0
        nt.links.new(bw.outputs["Val"], inv.inputs[1])
        gl = _node(nt, "ShaderNodeMath", operation="MULTIPLY")
        gl.inputs[1].default_value = pal["glow"]
        nt.links.new(inv.outputs["Value"], gl.inputs[0])
        bsdf.inputs["Emission Color"].default_value = (*pal.get("glow_color", pal["grout"])[:3], 1.0)
        nt.links.new(gl.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def horn_material(name, pal, r0=0.12, r1=0.5):
    """Horn/spike colour from root to tip (distance from the object origin), with growth rings (bump)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial")
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled")
    tc = _node(nt, "ShaderNodeTexCoord")
    ln = _node(nt, "ShaderNodeVectorMath", operation="LENGTH")
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    mr = _node(nt, "ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = r0
    mr.inputs["From Max"].default_value = r1
    nt.links.new(ln.outputs["Value"], mr.inputs["Value"])
    ramp = _ramp(nt, [(0.0, pal["horn_root"]), (0.55, pal["horn_mid"]), (1.0, pal["horn_tip"])])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    wave = _node(nt, "ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y")
    wave.inputs["Scale"].default_value = 55.0
    wave.inputs["Distortion"].default_value = 1.6
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    ring = _node(nt, "ShaderNodeMixRGB", blend_type="MULTIPLY")
    ring.inputs["Fac"].default_value = 0.10
    nt.links.new(ramp.outputs["Color"], ring.inputs["Color1"])
    nt.links.new(wave.outputs["Color"], ring.inputs["Color2"])
    nt.links.new(ring.outputs["Color"], bsdf.inputs["Base Color"])
    bp = _node(nt, "ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.15
    bp.inputs["Distance"].default_value = 0.003
    nt.links.new(wave.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = pal.get("horn_rough", 0.42)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m


def glow_eye_material(name, color, strength=2.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color[:3], 1.0)
    b.inputs["Roughness"].default_value = 0.15
    b.inputs["Emission Color"].default_value = (*color[:3], 1.0)
    b.inputs["Emission Strength"].default_value = strength
    return m


def set_grade(scene, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1), saturation=1.0, contrast=0.0):
    """Colour balance in the compositor: lift (shadows), gamma (mids), gain (highlights) per channel, then saturation
    and contrast. Rendering applies it (render.use_compositing)."""
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    cb = nt.nodes.new("CompositorNodeColorBalance")
    cb.correction_method = "LIFT_GAMMA_GAIN"
    cb.lift = tuple(1.0 + x for x in lift)
    cb.gamma = tuple(gamma)
    cb.gain = tuple(gain)
    hs = nt.nodes.new("CompositorNodeHueSat")
    hs.inputs["Saturation"].default_value = saturation
    bc = nt.nodes.new("CompositorNodeBrightContrast")
    bc.inputs["Contrast"].default_value = contrast
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], cb.inputs["Image"])
    nt.links.new(cb.outputs["Image"], hs.inputs["Image"])
    nt.links.new(hs.outputs["Image"], bc.inputs["Image"])
    nt.links.new(bc.outputs["Image"], comp.inputs["Image"])
    scene.render.use_compositing = True


# ----------------------------------------------------------------------------- Mirror modifier
def add_mirror(ob, axis=0, merge=True, clip=True, mirror_u=True, threshold=1e-5):
    """Model one half, let Blender's Mirror modifier produce the other (exact symmetry by construction)."""
    m = ob.modifiers.new("Mirror", "MIRROR")
    m.use_axis[axis] = True
    m.use_clip = clip
    m.use_mirror_merge = merge
    m.merge_threshold = threshold
    m.use_mirror_u = mirror_u
    return m


def apply_modifiers(ob):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


# ----------------------------------------------------------------------------- reference silhouettes
def load_rle_mask(d):
    """Decode the run-length silhouette stored in characters/dragon_ref/ref_silhouette.json."""
    m = np.zeros((d["h"], d["w"]), bool)
    for y, r in enumerate(d["rows"]):
        for s, e in zip(r[::2], r[1::2]):
            m[y, s:e] = True
    return m


def silhouette(objs, center, ppm, res, path, view="side"):
    """Orthographic silhouette of `objs` only, rendered with Workbench flat shading and a transparent film. Returns a
    bool mask (row 0 = top). The mapping is absolute (no registration): pixel (px, py) is
    h = ch + (px + 0.5 - w/2) / ppm, v = cv + (h/2 - py - 0.5) / ppm, with (h, v) =
    side (camera on +X): (y, z); front (camera on -Y, the nose side): (x, z); top (camera on +Z): (x, y)."""
    sc = bpy.context.scene
    w, h = res
    cy, cz = center
    keep = {o: o.hide_render for o in bpy.data.objects if o.type == "MESH"}
    for o in keep:
        o.hide_render = o not in objs
    if view == "side":
        cam = camera((3.0, cy, cz), (0.0, cy, cz), name="_SilCam", ortho=w / ppm)
    elif view == "front":
        cam = camera((cy, -3.0, cz), (cy, 0.0, cz), name="_SilCam", ortho=w / ppm)
    else:
        cam = camera((cy, cz, 3.0), (cy, cz, 0.0), name="_SilCam", ortho=w / ppm)
        cam.rotation_euler = (0.0, 0.0, 0.0)  # looking down -Z with +Y up in the image
    saved = (sc.camera, sc.render.engine, sc.render.film_transparent, sc.render.resolution_x, sc.render.resolution_y,
             sc.render.resolution_percentage, sc.render.image_settings.color_mode, sc.render.image_settings.file_format,
             sc.render.filepath, sc.display.shading.light, sc.display.shading.color_type, sc.display.render_aa)
    sc.camera = cam
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "SINGLE"
    sc.display.shading.single_color = (0, 0, 0)
    sc.display.render_aa = "OFF"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    a = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(a)
    mask = (a[3::4].reshape(h, w)[::-1]) > 0.5
    bpy.data.images.remove(img)
    d = cam.data
    bpy.data.objects.remove(cam)
    bpy.data.cameras.remove(d)
    for o, v in keep.items():
        o.hide_render = v
    (sc.camera, sc.render.engine, sc.render.film_transparent, sc.render.resolution_x, sc.render.resolution_y,
     sc.render.resolution_percentage, sc.render.image_settings.color_mode, sc.render.image_settings.file_format,
     sc.render.filepath, sc.display.shading.light, sc.display.shading.color_type, sc.display.render_aa) = saved
    return mask


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def outline(mask):
    """Per-column top and bottom row of a mask (NaN where the column is empty)."""
    h, w = mask.shape
    top = np.full(w, np.nan)
    bot = np.full(w, np.nan)
    for x in range(w):
        r = np.nonzero(mask[:, x])[0]
        if len(r):
            top[x], bot[x] = r[0], r[-1]
    return top, bot


def overlay_png(mine, ref, path):
    """Reference only = red, model only = cyan, both = light grey, empty = dark."""
    h, w = ref.shape
    img = np.ones((h, w, 3), np.float32) * np.array([0.05, 0.06, 0.08], np.float32)
    img[ref & ~mine] = (0.95, 0.15, 0.15)
    img[mine & ~ref] = (0.10, 0.85, 0.95)
    img[ref & mine] = (0.80, 0.80, 0.80)
    im = image_from_array("overlay", img[::-1], pack=False)
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)
