"""rblender PoC: build, texture, animate, render and export a scene entirely headless.

Usage: rblender demo [--engine cycles|eevee|workbench] [--quick]
"""
import math

import bpy
import rb

opts = rb.args()
engine = opts[opts.index("--engine") + 1] if "--engine" in opts else "CYCLES"
quick = "--quick" in opts
N_FRAMES = 12 if quick else 24

scene = rb.clean()
scene.frame_start, scene.frame_end = 1, N_FRAMES

# ---- texturing 1: a generated image texture (checker + gradient), saved and re-loaded as a file
SIZE = 256
img = bpy.data.images.new("checker", SIZE, SIZE, alpha=False)
px = []
for y in range(SIZE):
    for x in range(SIZE):
        c = 1.0 if ((x // 32) + (y // 32)) % 2 == 0 else 0.15
        px += [c * (x / SIZE), c * 0.4, c * (y / SIZE), 1.0]
img.pixels.foreach_set(px)
img.filepath_raw = rb.out("texture.png")
img.file_format = "PNG"
img.save()

mat = bpy.data.materials.new("Textured")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(rb.out("texture.png"))
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.35

# ---- texturing 2: fully procedural material (noise -> color ramp + bump)
pmat = bpy.data.materials.new("Procedural")
pmat.use_nodes = True
pt = pmat.node_tree
pb = pt.nodes["Principled BSDF"]
noise = pt.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 6.0
ramp = pt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.02, 0.1, 0.3, 1)
ramp.color_ramp.elements[1].color = (0.9, 0.7, 0.2, 1)
bump = pt.nodes.new("ShaderNodeBump")
pt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
pt.links.new(ramp.outputs["Color"], pb.inputs["Base Color"])
pt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
pt.links.new(bump.outputs["Normal"], pb.inputs["Normal"])

# ---- geometry
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 1), segments=48, ring_count=24)
ball = bpy.context.object
ball.name = "Ball"
bpy.ops.object.shade_smooth()
ball.data.materials.append(mat)

bpy.ops.mesh.primitive_cube_add(size=1.2, location=(2.2, 0, 0.6))
cube = bpy.context.object
cube.name = "Cube"
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project()  # UV unwrap
bpy.ops.object.mode_set(mode="OBJECT")
cube.data.materials.append(pmat)

bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
floor = bpy.context.object
floor.name = "Floor"
fm = bpy.data.materials.new("Floor")
fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
floor.data.materials.append(fm)

# ---- lights, camera, world
bpy.ops.object.light_add(type="SUN", location=(4, -4, 8))
sun = bpy.context.object
sun.data.energy = 4.0
sun.rotation_euler = (math.radians(50), 0, math.radians(35))
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.07, 0.1, 1)

bpy.ops.object.camera_add(location=(0.5, -7, 3))
cam = bpy.context.object
scene.camera = cam
bpy.ops.object.empty_add(location=(1.1, 0, 0.8))
target = bpy.context.object
con = cam.constraints.new("TRACK_TO")
con.target = target
con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"

# ---- animation: bouncing + spinning ball, cube rotating (keyframes, bezier interpolation)
for f in range(1, N_FRAMES + 1):
    t = (f - 1) / (N_FRAMES - 1)
    ball.location = (-1.5 + 3 * t, 0, 1 + abs(math.sin(t * math.pi * 2)) * 1.5)
    ball.rotation_euler = (0, t * math.tau, 0)
    ball.keyframe_insert("location", frame=f)
    ball.keyframe_insert("rotation_euler", frame=f)
    cube.rotation_euler = (0, 0, t * math.tau)
    cube.keyframe_insert("rotation_euler", frame=f)

# ---- render still + video, then export in several formats
mid = N_FRAMES // 2
still = rb.render_still("still.png", engine=engine, samples=16, res=(640, 360), frame=mid)
video = rb.render_anim("anim.mp4", engine=engine, samples=6, res=(480, 270), frames=(1, N_FRAMES))
exports = [rb.export(n) for n in ("scene.glb", "scene.fbx", "scene.obj", "scene.usdc", "scene.blend")]
rb.report(frames=N_FRAMES, objects=[o.name for o in bpy.data.objects], materials=[m.name for m in bpy.data.materials])
