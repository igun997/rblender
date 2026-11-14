# user script: arbitrary bpy, written into ./out via rb.out()
import bpy, rb
rb.clean()
bpy.ops.mesh.primitive_monkey_add()
bpy.ops.object.modifier_add(type="SUBSURF")
bpy.ops.object.shade_smooth()
bpy.ops.object.light_add(type="SUN", location=(3, -3, 5))
bpy.ops.object.camera_add(location=(0, -5, 1), rotation=(1.45, 0, 0))
bpy.context.scene.camera = bpy.context.object
rb.render_still("monkey_eevee.png", engine="EEVEE", samples=8, res=(320, 240))
rb.export("monkey.glb")
rb.report()
