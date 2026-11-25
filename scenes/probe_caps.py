"""Probe which sculpt/material/texture operations work in headless background Blender."""
import json
import math
import time
import traceback

import bmesh
import bpy
import rb
from mathutils import Vector

results = {}


def probe(name):
    def deco(fn):
        t = time.time()
        try:
            info = fn()
            results[name] = {"ok": True, "info": info, "s": round(time.time() - t, 2)}
        except Exception as e:  # noqa: BLE001
            results[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:160]}", "s": round(time.time() - t, 2)}
        return fn
    return deco


def sphere(subdiv=5, radius=1.0, name="S"):
    rb.clean()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=radius)
    ob = bpy.context.object
    ob.name = name
    return ob


def z_extent(ob):
    return max(v.co.z for v in ob.data.vertices)


@probe("sculpt.brush_stroke (bare, background)")
def _():
    ob = sphere()
    bpy.ops.object.mode_set(mode="SCULPT")
    before = z_extent(ob)
    stroke = [{"name": "s", "location": (0, 0, 1.0), "mouse": (0, 0), "mouse_event": (0, 0), "pressure": 1.0,
               "size": 50, "pen_flip": False, "x_tilt": 0, "y_tilt": 0, "time": 0.0, "is_start": True}]
    bpy.ops.sculpt.brush_stroke(stroke=stroke, mode="NORMAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"dz": round(z_extent(ob) - before, 5)}


@probe("sculpt mode enter + brush datablock")
def _():
    ob = sphere()
    bpy.ops.object.mode_set(mode="SCULPT")
    ts = bpy.context.tool_settings
    b = ts.sculpt.brush
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"brush": b.name if b else None}


@probe("object.voxel_remesh")
def _():
    ob = sphere(subdiv=3)
    ob.data.remesh_voxel_size = 0.08
    n0 = len(ob.data.vertices)
    bpy.ops.object.voxel_remesh()
    return {"verts": [n0, len(ob.data.vertices)]}


@probe("Remesh modifier (VOXEL) applied")
def _():
    ob = sphere(subdiv=3)
    m = ob.modifiers.new("r", "REMESH")
    m.mode = "VOXEL"
    m.voxel_size = 0.07
    bpy.ops.object.modifier_apply(modifier="r")
    return {"verts": len(ob.data.vertices)}


@probe("Multires subdivide + sculpt levels")
def _():
    ob = sphere(subdiv=2)
    m = ob.modifiers.new("mr", "MULTIRES")
    for _ in range(3):
        bpy.ops.object.multires_subdivide(modifier="mr", mode="CATMULL_CLARK")
    return {"levels": m.total_levels}


@probe("Dyntopo toggle (sculpt.dynamic_topology_toggle)")
def _():
    ob = sphere()
    bpy.ops.object.mode_set(mode="SCULPT")
    bpy.ops.sculpt.dynamic_topology_toggle()
    on = ob.use_dynamic_topology_sculpting if hasattr(ob, "use_dynamic_topology_sculpting") else None
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"enabled": on}


@probe("Shrinkwrap modifier (project onto reference)")
def _():
    ref = sphere(subdiv=4, radius=1.0, name="Ref")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, segments=64, ring_count=32)
    tgt = bpy.context.object
    m = tgt.modifiers.new("sw", "SHRINKWRAP")
    m.target = ref
    m.wrap_method = "NEAREST_SURFACEPOINT"
    dg = bpy.context.evaluated_depsgraph_get()
    ev = tgt.evaluated_get(dg)
    r = max(v.co.length for v in ev.data.vertices)
    return {"max_radius_after": round(r, 4)}


@probe("Decimate (collapse) modifier")
def _():
    ob = sphere(subdiv=6)
    n0 = len(ob.data.polygons)
    m = ob.modifiers.new("d", "DECIMATE")
    m.ratio = 0.1
    bpy.ops.object.modifier_apply(modifier="d")
    return {"faces": [n0, len(ob.data.polygons)]}


@probe("Displace modifier with procedural texture")
def _():
    ob = sphere(subdiv=5)
    tex = bpy.data.textures.new("n", "CLOUDS")
    m = ob.modifiers.new("disp", "DISPLACE")
    m.texture = tex
    m.strength = 0.2
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    return {"max_r": round(max(v.co.length for v in ev.data.vertices), 3)}


@probe("Cycles bake NORMAL high->low (selected_to_active)")
def _():
    rb.clean()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.0)
    hi = bpy.context.object
    tex = bpy.data.textures.new("n", "CLOUDS")
    d = hi.modifiers.new("d", "DISPLACE")
    d.texture, d.strength = tex, 0.08
    bpy.ops.object.modifier_apply(modifier="d")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
    lo = bpy.context.object
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    img = bpy.data.images.new("bake", 256, 256)
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    mat.node_tree.nodes.active = node
    lo.data.materials.append(mat)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 1
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = 0.15
    sc.render.bake.margin = 4
    for o in bpy.context.selected_objects:
        o.select_set(False)
    hi.select_set(True)
    lo.select_set(True)
    bpy.context.view_layer.objects.active = lo
    bpy.ops.object.bake(type="NORMAL")
    px = list(img.pixels)
    spread = max(px[0::4]) - min(px[0::4])
    return {"red_spread": round(spread, 3)}


@probe("Cycles bake AO + diffuse color from procedural")
def _():
    ob = sphere(subdiv=4)
    ob.data.uv_layers.new(name="UV")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    nt.links.new(noise.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    img = bpy.data.images.new("bk", 128, 128)
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    nt.nodes.active = node
    ob.data.materials.append(mat)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 4
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    px = list(img.pixels)
    return {"red_spread": round(max(px[0::4]) - min(px[0::4]), 3)}


@probe("Smart UV project + texel density readout")
def _():
    ob = sphere(subdiv=4)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66))
    bpy.ops.object.mode_set(mode="OBJECT")
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    uv = bm.loops.layers.uv.active
    ua = sum(abs((f.loops[1][uv].uv - f.loops[0][uv].uv).cross(f.loops[2][uv].uv - f.loops[0][uv].uv)) / 2 for f in bm.faces)
    ga = sum(f.calc_area() for f in bm.faces)
    bm.free()
    return {"uv_area": round(ua, 3), "surf_area": round(ga, 3)}


@probe("Image paint by pixel buffer (numpy brush into texture)")
def _():
    import numpy as np
    img = bpy.data.images.new("paint", 128, 128)
    buf = np.zeros((128, 128, 4), np.float32)
    buf[..., 3] = 1
    yy, xx = np.mgrid[0:128, 0:128]
    d = np.hypot(xx - 64, yy - 64)
    buf[..., 0] = np.clip(1 - d / 30, 0, 1)
    img.pixels.foreach_set(buf.ravel())
    return {"max_red": round(float(buf[..., 0].max()), 2)}


@probe("texture_paint mode + paint.image_paint stroke")
def _():
    ob = sphere(subdiv=3)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"entered": True}


@probe("BVH ray cast / nearest (measurement)")
def _():
    from mathutils.bvhtree import BVHTree
    ob = sphere(subdiv=4, radius=2.0)
    t = BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())
    loc, nor, idx, dist = t.ray_cast(Vector((0, 0, 5)), Vector((0, 0, -1)))
    return {"hit_z": round(loc.z, 3)}


@probe("Geometry Nodes group via Python")
def _():
    ob = sphere(subdiv=3)
    ng = bpy.data.node_groups.new("gn", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    i, o = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    sub = ng.nodes.new("GeometryNodeSubdivisionSurface")
    ng.links.new(i.outputs[0], sub.inputs[0])
    ng.links.new(sub.outputs[0], o.inputs[0])
    m = ob.modifiers.new("gn", "NODES")
    m.node_group = ng
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {"verts": len(ev.data.vertices)}


@probe("Sculpt-mode-free: sculpt.mask / face set ops via attributes")
def _():
    ob = sphere(subdiv=3)
    ob.data.attributes.new(".sculpt_mask", "FLOAT", "POINT")
    return {"attrs": len(ob.data.attributes)}


print("PROBE_RESULTS " + json.dumps(results))
with open(rb.out("probe_caps.json"), "w") as fh:
    json.dump({"blender": bpy.app.version_string, "results": results}, fh, indent=2)
