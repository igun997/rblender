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
ms = [o for o in bpy.data.objects if o.type == "MESH"]
P = np.vstack([np.array([list(o.matrix_world @ v.co) for v in o.data.vertices]) for o in ms])
print("REIMPORT " + json.dumps({
    "meshes": len(ms), "verts": len(P), "tris": sum(len(p.vertices) - 2 for o in ms for p in o.data.polygons),
    "materials": len(bpy.data.materials), "armatures": sum(1 for o in bpy.data.objects if o.type == "ARMATURE"),
    "x_range_m": [round(float(P[:, 0].min()), 3), round(float(P[:, 0].max()), 3)],
    "size_m": [round(float(v), 3) for v in P.max(0) - P.min(0)]}))
