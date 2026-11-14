import bpy, os, json
bpy.ops.wm.read_factory_settings(use_empty=True)
p = os.path.join(os.environ["RBLENDER_OUT"], "vestia_zeta.glb")
bpy.ops.import_scene.gltf(filepath=p)
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
a = arms[0]
act = a.animation_data.action if a.animation_data else None
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()
h1 = (a.matrix_world @ a.pose.bones["hand.L"].matrix).translation.copy() if "hand.L" in a.pose.bones else None
bpy.context.scene.frame_set(54)
bpy.context.view_layer.update()
h54 = (a.matrix_world @ a.pose.bones["hand.L"].matrix).translation.copy() if "hand.L" in a.pose.bones else None
skinned = sum(1 for m in meshes if any(md.type == "ARMATURE" for md in m.modifiers))
print("REIMPORT", json.dumps({
  "armatures": len(arms), "bones": len(a.data.bones), "meshes": len(meshes), "skinned_meshes": skinned,
  "action": act.name if act else None, "frame_range": tuple(act.frame_range) if act else None,
  "hand_L_frame1": [round(x, 3) for x in h1] if h1 else None, "hand_L_frame54_salute": [round(x, 3) for x in h54] if h54 else None,
  "images": len(bpy.data.images), "materials": len(bpy.data.materials)}))
