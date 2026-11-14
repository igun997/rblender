"""Bone registry. Geometry modules call add_bone() as they create things that need to move;
rig.py turns the registry into an armature. Positions are armature (== world) space at rest."""
from mathutils import Vector as V

from . import layout as L

BONES = {}  # name -> dict(head, tail, parent)
ORDER = []


def add_bone(name, head, tail, parent=None):
    if name in BONES:
        return name
    BONES[name] = {"head": V(head), "tail": V(tail), "parent": parent}
    ORDER.append(name)
    return name


def seg(name):
    b = BONES[name]
    return b["head"], b["tail"]


def segments(names):
    return {n: (BONES[n]["head"], BONES[n]["tail"]) for n in names}


FINGERS = ("index", "middle", "ring", "pinky", "thumb")


def build_base():
    add_bone("root", (0, 0, 0.0), (0, 0, 0.12))
    add_bone("hips", (0, 0, 0.88), (0, 0, 0.96), "root")
    add_bone("spine", (0, 0, 0.96), (0, 0, 1.10), "hips")
    add_bone("chest", (0, 0, 1.10), (0, 0, 1.25), "spine")
    add_bone("neck", (0, 0.008, 1.285), (0, 0.012, 1.35), "chest")
    add_bone("head", (0, 0.012, 1.35), (0, 0.012, 1.57), "neck")
    for sd, s in L.SIDES.items():
        a = L.arm_pts(s)
        g = L.hand_geometry(s)
        add_bone(f"clavicle.{sd}", a["clav"], a["shoulder"], "chest")
        add_bone(f"upper_arm.{sd}", a["shoulder"], a["elbow"], f"clavicle.{sd}")
        add_bone(f"forearm.{sd}", a["elbow"], a["wrist"], f"upper_arm.{sd}")
        add_bone(f"hand.{sd}", a["wrist"], g["knuckle"], f"forearm.{sd}")
        for fn in FINGERS:
            pts = g["fingers"][fn]
            parent = f"hand.{sd}"
            for k in range(3):
                nm = f"{fn}{k + 1}.{sd}"
                add_bone(nm, pts[k], pts[k + 1], parent)
                parent = nm
        lp = L.leg_pts(s)
        add_bone(f"thigh.{sd}", lp["hip"], lp["knee"], "hips")
        add_bone(f"shin.{sd}", lp["knee"], lp["ankle"], f"thigh.{sd}")
        add_bone(f"foot.{sd}", lp["ankle"], lp["ball"], f"shin.{sd}")
        add_bone(f"toe.{sd}", lp["ball"], lp["toe"], f"foot.{sd}")


def arm_chain(sd):
    return [f"clavicle.{sd}", f"upper_arm.{sd}", f"forearm.{sd}", f"hand.{sd}"]


def finger_bones(sd):
    return [f"{fn}{k}.{sd}" for fn in FINGERS for k in (1, 2, 3)]
