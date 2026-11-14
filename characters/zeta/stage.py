"""Studio: world, lights, cameras, and render look (PBR or toon)."""
import math

import bpy
from mathutils import Vector as V


def world(color_top=(0.20, 0.24, 0.34), color_bottom=(0.06, 0.07, 0.10)):
    w = bpy.data.worlds.new("Studio")
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mapr = nt.nodes.new("ShaderNodeMapRange")
    mapr.inputs["From Min"].default_value = -0.3
    mapr.inputs["From Max"].default_value = 0.9
    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mapr.inputs["Value"])
    nt.links.new(mapr.outputs["Result"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].color = (*color_bottom, 1)
    ramp.color_ramp.elements[1].color = (*color_top, 1)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = w
    return w


def floor(M_mat=None, size=8.0):
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=size / 2, fill_type="NGON", location=(0, 0, 0))
    ob = bpy.context.object
    ob.name = "StudioFloor"
    m = bpy.data.materials.new("StudioFloor")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.55, 0.57, 0.62, 1)
    b.inputs["Roughness"].default_value = 0.85
    ob.data.materials.append(m)
    return ob


def lights(strength=1.0):
    def area(name, loc, energy, size, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "AREA")
        d.energy, d.size, d.color = energy * strength, size, color
        o = bpy.data.objects.new(name, d)
        bpy.context.scene.collection.objects.link(o)
        o.location = loc
        direction = V((0, 0, 1.0)) - V(loc)
        o.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return o
    area("Key", (1.6, -2.4, 2.4), 150, 1.6, (1.0, 0.96, 0.92))
    area("Fill", (-2.2, -1.8, 1.4), 60, 2.0, (0.85, 0.9, 1.0))
    area("Rim", (-0.6, 2.2, 2.3), 140, 1.2, (0.8, 0.88, 1.0))


def camera(loc, target, lens=70, name="Cam"):
    d = bpy.data.cameras.new(name)
    d.lens = lens
    d.clip_start, d.clip_end = 0.05, 50
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (V(target) - V(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def orbit(angle_deg, dist=4.6, height=1.15, target=(0, 0, 0.82), lens=70, name=None):
    a = math.radians(angle_deg)
    loc = (dist * math.sin(a), -dist * math.cos(a), height)
    return camera(loc, target, lens, name or f"Cam_{int(angle_deg)}")


def look(scene, exposure=0.0):
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = exposure - 0.35
    ee = scene.eevee
    for attr, val in (("shadow_ray_count", 3), ("shadow_step_count", 12)):
        if hasattr(ee, attr):
            setattr(ee, attr, val)


def contact_sheet(paths, out_path, pad=6, bg=(0.04, 0.05, 0.07)):
    """Tile rendered PNGs left-to-right into one image (numpy), saved to out_path."""
    import numpy as np
    from .common import image_from_array
    tiles = []
    for p in paths:
        im = bpy.data.images.load(p)
        w, h = im.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(buf)
        tiles.append(buf.reshape(h, w, 4)[..., :3])
        bpy.data.images.remove(im)
    h = max(t.shape[0] for t in tiles)
    row = []
    for t in tiles:
        canvas = np.ones((h, t.shape[1], 3), np.float32) * np.array(bg, np.float32)
        canvas[: t.shape[0]] = t
        row.append(canvas)
        row.append(np.ones((h, pad, 3), np.float32) * np.array(bg, np.float32))
    sheet = np.concatenate(row[:-1], axis=1)
    image_from_array("sheet", sheet, path=out_path)


def wireframe_render(out_path, res=(540, 960), samples=8):
    """Topology proof: Cycles with a wireframe-node material override on every object."""
    scene = bpy.context.scene
    m = bpy.data.materials.new("WireOverride")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    wire = nt.nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 0.9
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.02, 0.03, 0.05, 1)
    mix.inputs["B"].default_value = (0.55, 0.85, 1.0, 1)
    emi = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(wire.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    vl = bpy.context.view_layer
    vl.material_override = m
    floor_ob = bpy.data.objects.get("StudioFloor")
    if floor_ob:
        floor_ob.hide_render = True
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.device = "CPU"
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_path
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)
    vl.material_override = None
    if floor_ob:
        floor_ob.hide_render = False
