"""Turn the bone registry into an armature and bind every mesh to it."""
import bpy
from mathutils import Vector as V

from . import skeleton as SK
from .common import OBJECTS


def build_armature(name="Zeta_Rig"):
    arm = bpy.data.armatures.new("ZetaArmature")
    rig = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    for nm in SK.ORDER:
        b = SK.BONES[nm]
        e = eb.new(nm)
        e.head, e.tail = b["head"], b["tail"]
        if (e.tail - e.head).length < 1e-4:
            e.tail = e.head + V((0, 0, 0.01))
        if b["parent"]:
            e.parent = eb[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.display_type = "STICK"
    rig.show_in_front = True
    # bone collections for tidy export/editing
    groups = {
        "Body": lambda n: n in ("root", "hips", "spine", "chest", "neck", "head") or n.startswith(("clavicle", "upper_arm", "forearm", "thigh", "shin", "foot", "toe")),
        "Hands": lambda n: n.startswith(("hand", "index", "middle", "ring", "pinky", "thumb")),
        "Hair": lambda n: n.startswith(("hair", "tuft", "ribbon")),
        "Cloth": lambda n: n.startswith(("skirt", "cape")),
        "Face": lambda n: n.startswith(("eye", "mouth")),
    }
    for gname, pred in groups.items():
        col = arm.collections.new(gname)
        for b in arm.bones:
            if pred(b.name):
                col.assign(b)
    return rig


def bind(rig):
    """Parent every mesh to the rig with an Armature modifier. Returns (bound, unweighted_vertices)."""
    bound, bad = 0, 0
    for ob in list(OBJECTS.values()):
        if ob is None or ob.name not in bpy.data.objects or ob.type != "MESH" or ob is rig:
            continue
        if not ob.vertex_groups:
            continue
        ob.parent = rig
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        bound += 1
        for v in ob.data.vertices:
            if not any(g.weight > 0 for g in v.groups):
                bad += 1
    return bound, bad
