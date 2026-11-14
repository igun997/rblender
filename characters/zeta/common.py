"""Shared modeling helpers: lofted meshes, surface-projected decals, procedural textures, skin weights.

World convention: Z up, character faces -Y, character's left is +X (so `.L` == +X).
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

V = Vector
OBJECTS = {}  # name -> bpy.types.Object, everything the build creates (for rigging/export)
RIGID = {}  # object name -> bone name for rigid (non-deformed) parts


# --------------------------------------------------------------------------- materials / textures
def image_from_array(name, arr, path=None):
    """arr: float array (h, w, 3|4) in 0..1, row 0 at the BOTTOM (Blender convention)."""
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), arr.dtype)], axis=2)
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.pixels.foreach_set(np.clip(arr, 0, 1).astype(np.float32).ravel())
    if path:
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        img = bpy.data.images.load(path)
        img.name = name
    img.pack()
    return img


def material(name, color=(0.8, 0.8, 0.8), rough=0.55, metal=0.0, tex=None, emission=None, spec=0.4):
    """Principled material. `tex` (Image) drives Base Color. Kept exporter-friendly (glTF)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color[:3], 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = spec
    if tex is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = tex
        t.interpolation = "Linear"
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if emission:
        b.inputs["Emission Color"].default_value = (*emission[:3], 1)
        b.inputs["Emission Strength"].default_value = 1.0
    m["toon_color"] = list(color[:3])
    return m


# --------------------------------------------------------------------------- mesh creation
def new_object(name, bm, mat=None, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat is not None:
        for m in mat if isinstance(mat, (list, tuple)) else [mat]:
            me.materials.append(m)
    if smooth:
        me.shade_smooth()
    OBJECTS[name] = ob
    return ob


def _orient_outward(bm):
    c = sum((v.co for v in bm.verts), V()) / max(len(bm.verts), 1)
    s = sum(f.normal.dot(f.calc_center_median() - c) * f.calc_area() for f in bm.faces)
    if s < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])


def loft_bm(bm, centers, rx, ry, n=24, side=(1, 0, 0), caps=(True, True), radial=None,
            uv=False, twist=None, outward=True):
    """Add a lofted tube to `bm`. centers: list of points; rx/ry: scalar or per-ring list.
    radial(i, theta) -> scale multiplier. Ring local x = `side` projected orthogonal to the path."""
    N = len(centers)
    cs = [V(c) for c in centers]
    rx = [rx] * N if not isinstance(rx, (list, tuple)) else list(rx)
    ry = [ry] * N if not isinstance(ry, (list, tuple)) else list(ry)
    side = V(side)
    uvl = bm.loops.layers.uv.verify() if uv else None
    rings = []
    prev_s = None
    for i, c in enumerate(cs):
        t = (cs[min(i + 1, N - 1)] - cs[max(i - 1, 0)])
        if t.length < 1e-9:
            t = V((0, 0, -1))
        t.normalize()
        ref = side if prev_s is None else prev_s
        s = ref - t * ref.dot(t)
        if s.length < 1e-6:
            s = V((1, 0, 0)) - t * t.x
        s.normalize()
        prev_s = s
        u = t.cross(s)
        ring = []
        for j in range(n):
            th = math.tau * j / n
            k = radial(i, th) if radial else 1.0
            ring.append(bm.verts.new(c + s * (rx[i] * k * math.cos(th)) + u * (ry[i] * k * math.sin(th))))
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
    if caps[0]:
        made.append(bm.faces.new(rings[0][::-1]))
    if caps[1]:
        made.append(bm.faces.new(rings[-1]))
    if outward:
        # orient just this tube: temporarily gather its faces
        c = sum((v.co for r in rings for v in r), V()) / (N * n)
        s = sum(f.normal.dot(f.calc_center_median() - c) * f.calc_area() for f in made)
        if s < 0:
            bmesh.ops.reverse_faces(bm, faces=made)
    return rings


def loft(name, centers, rx, ry, mat=None, smooth=True, **kw):
    bm = bmesh.new()
    loft_bm(bm, centers, rx, ry, **kw)
    return new_object(name, bm, mat, smooth)


def lerp_path(points, count):
    """Resample a polyline into `count` evenly spaced points."""
    pts = [V(p) for p in points]
    seg = [(pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)]
    total = sum(seg)
    out = []
    for k in range(count):
        d = total * k / (count - 1)
        i = 0
        while i < len(seg) - 1 and d > seg[i]:
            d -= seg[i]
            i += 1
        t = 0 if seg[i] == 0 else min(d / seg[i], 1.0)
        out.append(pts[i].lerp(pts[i + 1], t))
    return out


def _face_to(bm, direction):
    """Make an open sheet's normals point against `direction` (i.e. toward the viewer/outside)."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    d = V(direction)
    if sum(f.normal.dot(d) for f in bm.faces) > 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])


def catmull(points, per=6):
    """Smooth a polyline with Catmull-Rom splines."""
    p = [V(q) for q in points]
    if len(p) < 3:
        return p
    ext = [p[0] * 2 - p[1]] + p + [p[-1] * 2 - p[-2]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(per):
            t = k / per
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1])
    return out


def smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def box(name, center, size, mat=None, rot=(0, 0, 0), bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = V((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2, affect="EDGES")
    R = Matrix.Rotation(rot[2], 4, "Z") @ Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ R, verts=bm.verts)
    return new_object(name, bm, mat, smooth=bool(bevel))


def flat_shape(name, pts2d, plane_pt, ax_u, ax_v, thickness=0.002, mat=None):
    """Extruded 2D polygon (for stars, clips, buckles). pts2d in (u,v) units along ax_u/ax_v."""
    bm = bmesh.new()
    ax_u, ax_v = V(ax_u), V(ax_v)
    n = ax_u.cross(ax_v).normalized()
    top = [bm.verts.new(V(plane_pt) + ax_u * u + ax_v * v + n * thickness / 2) for u, v in pts2d]
    bot = [bm.verts.new(V(plane_pt) + ax_u * u + ax_v * v - n * thickness / 2) for u, v in pts2d]
    bm.faces.new(top)
    bm.faces.new(bot[::-1])
    k = len(pts2d)
    for i in range(k):
        bm.faces.new((top[i], bot[i], bot[(i + 1) % k], top[(i + 1) % k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return new_object(name, bm, mat, smooth=False)


def star(name, center, ax_u, ax_v, r_out, r_in, mat, points=4, thickness=0.002):
    pts = []
    for k in range(points * 2):
        r = r_out if k % 2 == 0 else r_in
        a = math.pi * k / points
        pts.append((r * math.cos(a), r * math.sin(a)))
    return flat_shape(name, pts, center, ax_u, ax_v, thickness, mat)


# --------------------------------------------------------------------------- surface projection / decals
class Surface:
    """Ray-cast helper onto an object (world == local; objects are never transformed)."""

    def __init__(self, ob):
        dg = bpy.context.evaluated_depsgraph_get()
        self.tree = BVHTree.FromObject(ob, dg)

    def hit(self, origin, direction):
        loc, nor, idx, dist = self.tree.ray_cast(V(origin), V(direction).normalized())
        return (loc, nor) if loc is not None else (None, None)

    def project(self, p, direction=(0, 1, 0), offset=0.0, back=0.6):
        """Project p along `direction` onto the surface (starting `back` m behind p)."""
        d = V(direction).normalized()
        loc, nor = self.hit(V(p) - d * back, d)
        if loc is None:  # ray missed: snap to the closest point instead of leaving a stray vertex
            loc, nor, _idx, _dist = self.tree.find_nearest(V(p))
            if loc is None:
                return None
        return loc + nor * offset


def decal(name, target_ob, uv_fn, shape, mat, nu=10, nv=10, offset=0.0015, direction=(0, 1, 0),
          uv=True, surface=None):
    """Grid sheet whose vertices are projected onto `target_ob`.
    uv_fn(u, v) -> world point in front of the surface (u, v in 0..1), projected along `direction`.
    shape(u, v) -> bool keep-vertex mask for non-rect decals (cells touching removed verts are skipped)."""
    sf = surface or Surface(target_ob)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify() if uv else None
    grid = {}
    for i in range(nu + 1):
        for j in range(nv + 1):
            u, v = i / nu, j / nv
            if shape and not shape(u, v):
                continue
            p = sf.project(uv_fn(u, v), direction, offset)
            if p is not None:
                grid[(i, j)] = bm.verts.new(p)
    for i in range(nu):
        for j in range(nv):
            quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(q in grid for q in quad):
                f = bm.faces.new([grid[q] for q in quad])
                if uv:
                    for loop, (a, b) in zip(f.loops, quad):
                        loop[uvl].uv = (a / nu, b / nv)
    if not bm.faces:
        bm.free()
        return None
    _face_to(bm, direction)
    return new_object(name, bm, mat, smooth=True)


def ribbon_decal(name, target_ob, pts, widths, mat, direction=(0, 1, 0), offset=0.002, steps=24,
                 surface=None, across=2):
    """Strip following polyline `pts` (3D points in front of the surface) with per-point widths."""
    sf = surface or Surface(target_ob)
    path = catmull(pts, 8) if len(pts) > 2 else lerp_path(pts, steps)
    path = lerp_path(path, steps)
    w = np.interp(np.linspace(0, 1, steps), np.linspace(0, 1, len(widths)), widths)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    rows = []
    for i, p in enumerate(path):
        t = (path[min(i + 1, steps - 1)] - path[max(i - 1, 0)]).normalized()
        d = V(direction).normalized()
        s = t.cross(d).normalized()
        row = []
        for k in range(across + 1):
            q = p + s * (w[i] * (k / across - 0.5))
            hp = sf.project(q, direction, offset)
            row.append(bm.verts.new(hp if hp is not None else q))
        rows.append(row)
    for i in range(steps - 1):
        for k in range(across):
            f = bm.faces.new((rows[i][k], rows[i][k + 1], rows[i + 1][k + 1], rows[i + 1][k]))
            for loop, (ii, kk) in zip(f.loops, ((i, k), (i, k + 1), (i + 1, k + 1), (i + 1, k))):
                loop[uvl].uv = (kk / across, ii / (steps - 1))
    _face_to(bm, direction)
    return new_object(name, bm, mat, smooth=True)


# --------------------------------------------------------------------------- weights
def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (p - (a + ab * t)).length


def skin_weights(ob, bones, restrict, k=4.0, maxinf=4, bias=None):
    """Distance based weights. bones: name -> (head, tail). restrict: bone names allowed on this mesh.
    bias: optional name -> multiplier."""
    segs = {n: (V(bones[n][0]), V(bones[n][1])) for n in restrict}
    groups = {n: ob.vertex_groups.new(name=n) for n in restrict}
    for v in ob.data.vertices:
        ds = []
        for n, (a, b) in segs.items():
            d = max(seg_dist(v.co, a, b), 1e-4)
            w = 1.0 / d ** k
            if bias and n in bias:
                w *= bias[n]
            ds.append((w, n))
        ds.sort(reverse=True)
        top = ds[:maxinf]
        tot = sum(w for w, _ in top)
        for w, n in top:
            if w / tot > 0.01:
                groups[n].add([v.index], w / tot, "REPLACE")


def rigid_weights(ob, bone):
    g = ob.vertex_groups.new(name=bone)
    g.add([v.index for v in ob.data.vertices], 1.0, "REPLACE")


def join(name, objs, mat_order=None):
    """Join objects (keeping all material slots) into one new object named `name`."""
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    ob = bpy.context.object
    ob.name = name
    ob.data.name = name
    for k in [k for k in OBJECTS if k not in bpy.data.objects or bpy.data.objects[k] is not OBJECTS[k]]:
        OBJECTS.pop(k, None)
    OBJECTS[name] = ob
    ob.select_set(False)
    return ob


def disc_decal(name, target_ob, center, width, height, mat, rot=0.0, rings=6, segs=28, offset=0.0015,
               direction=(0, 1, 0), surface=None, plane=(0, 2), shade=None):
    """Elliptical decal (eyes, open mouth). `center` is a point in front of the surface; the ellipse
    lies in the XZ plane (plane=(0,2)) and is projected along `direction`. UV spans the bounding box."""
    sf = surface or Surface(target_ob)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    cr, sr = math.cos(rot), math.sin(rot)
    center = V(center)
    verts = {}
    verts[(0, 0)] = None
    pts = {}

    def place(r, k):
        th = math.tau * k / segs
        lx, lz = 0.5 * width * r * math.cos(th), 0.5 * height * r * math.sin(th)
        q = center + V((lx * cr - lz * sr, 0, lx * sr + lz * cr))
        hit = sf.project(q, direction, offset)
        return bm.verts.new(hit if hit is not None else q), (0.5 + 0.5 * r * math.cos(th), 0.5 + 0.5 * r * math.sin(th))

    c, cuv = place(0.0, 0)
    ring_v = []
    for ri in range(1, rings + 1):
        r = ri / rings
        ring_v.append([place(r, k) for k in range(segs)])
    for k in range(segs):
        f = bm.faces.new((c, ring_v[0][(k + 1) % segs][0], ring_v[0][k][0]))
        for loop, uvv in zip(f.loops, (cuv, ring_v[0][(k + 1) % segs][1], ring_v[0][k][1])):
            loop[uvl].uv = uvv
    for ri in range(rings - 1):
        for k in range(segs):
            a, b = ring_v[ri][k], ring_v[ri][(k + 1) % segs]
            c2, d = ring_v[ri + 1][(k + 1) % segs], ring_v[ri + 1][k]
            f = bm.faces.new((a[0], b[0], c2[0], d[0]))
            for loop, uvv in zip(f.loops, (a[1], b[1], c2[1], d[1])):
                loop[uvl].uv = uvv
    _face_to(bm, direction)
    return new_object(name, bm, mat, smooth=True)
