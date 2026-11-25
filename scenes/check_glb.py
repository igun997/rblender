"""Import a glb into an empty scene and report what survived: meshes, vertices, triangles, materials, x extent.

    rblender run scenes/check_glb.py -- exports/dragon_ref3d.glb
"""
import json

import bpy
import numpy as np
import rb

path = next(a for a in rb.args() if a.endswith((".glb", ".gltf")))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
ms = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("Icosphere")]  # Icosphere = the importer's bone shape
P = np.vstack([np.array([list(o.matrix_world @ v.co) for v in o.data.vertices]) for o in ms])
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
acts = [(a.name, [round(float(x)) for x in a.frame_range]) for a in bpy.data.actions]
ext = {o.name: round(float(np.ptp(np.array([(o.matrix_world @ v.co).x for v in o.data.vertices]))), 3) for o in ms}
print("REIMPORT " + json.dumps({
    "meshes": len(ms), "verts": len(P), "tris": sum(len(p.vertices) - 2 for o in ms for p in o.data.polygons),
    "materials": len(bpy.data.materials), "armatures": len(arms), "bones": sum(len(a.data.bones) for a in arms),
    "skinned_meshes": sum(1 for o in ms if any(m.type == "ARMATURE" for m in o.modifiers)), "actions": acts,
    "x_range_m": [round(float(P[:, 0].min()), 3), round(float(P[:, 0].max()), 3)],
    "size_m": [round(float(v), 3) for v in P.max(0) - P.min(0)],
    "widest_x": sorted(ext.items(), key=lambda kv: -kv[1])[:3]}))
