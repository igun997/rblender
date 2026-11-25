"""Turn a rigged dragon glb into a segmented head target (numpy only, no Blender, no network).

    python3 characters/dragon_ref/prep_glb.py refs3d/red_dragon.glb refs3d/red_dragon_head.npz

The skin weights decide which part every vertex belongs to, so the split is exact instead of a guess from geometry:
    upper  head, jowlTop, lipUpper, eyelid, neck (the back of the skull)
    jaw    jaw, jowlBot, lipLower
    horn   horn*.L / horn*.R  (kept apart: one label per side)
    tongue tongue*
Raw bind-pose positions are used (the glb's own frame, nose towards -z). Only the head region is written.
The source model is third-party (CC-BY); the glb and the npz stay out of the repo, only derived numbers go in.
"""
import json
import struct
import sys

import numpy as np

LABELS = ["upper", "jaw", "horn.L", "horn.R", "tongue", "other"]
REGION_Z = -5.0   # head: z below this (nose at -8.56 in the glb frame)
REGION_Y = 1.4    # and above this (drops forelegs)


def read_glb(path):
    b = open(path, "rb").read()
    n, _ = struct.unpack("<II", b[12:20])
    return json.loads(b[20:20 + n]), b[20 + n + 8:]


def accessor(j, buf, i):
    a = j["accessors"][i]
    bv = j["bufferViews"][a["bufferView"]]
    comp = {5126: "f4", 5123: "u2", 5125: "u4", 5121: "u1"}[a["componentType"]]
    k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    isz = np.dtype(comp).itemsize * k
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride")
    if stride and stride != isz:
        arr = np.frombuffer(buf, np.uint8, a["count"] * stride, off).reshape(a["count"], stride)[:, :isz].copy().view(comp)
    else:
        arr = np.frombuffer(buf, comp, a["count"] * k, off)
    arr = arr.reshape(a["count"], k) if k > 1 else arr
    if a.get("normalized"):
        arr = arr / {5121: 255.0, 5123: 65535.0}[a["componentType"]]
    return arr


def group(name):
    n = name.lower()
    side = "L" if ".l" in n else "R" if ".r" in n else ""
    if n.startswith(("head", "jowltop", "lipupper", "eyelid", "neck")):
        return "upper"
    if n.startswith(("jaw", "jowlbot", "liplower")):
        return "jaw"
    if n.startswith("horn"):
        return "horn." + side
    if n.startswith("tongue"):
        return "tongue"
    return "other"


def labels_of(j, buf, prim, bone_group):
    at = prim["attributes"]
    jn = accessor(j, buf, at["JOINTS_0"]).astype(int)
    w = accessor(j, buf, at["WEIGHTS_0"]).astype(float)
    acc = np.zeros((len(jn), len(LABELS)))
    for k in range(4):
        for gi, g in enumerate(LABELS):
            acc[:, gi] += w[:, k] * (bone_group[jn[:, k]] == g)
    return acc.argmax(1)


def main(src, dst):
    j, buf = read_glb(src)
    names = [j["nodes"][i]["name"] for i in j["skins"][0]["joints"]]
    bone_group = np.array([group(n) for n in names])
    out = {}
    for m in j["meshes"]:
        prim = m["primitives"][0]
        key = m["name"].split("_")[1]  # mesh / eyes / teeth
        P = accessor(j, buf, prim["attributes"]["POSITION"]).astype(float)
        T = accessor(j, buf, prim["indices"]).astype(int).reshape(-1, 3)
        lab = labels_of(j, buf, prim, bone_group) if "JOINTS_0" in prim["attributes"] else np.full(len(P), 5)
        if key == "mesh":
            keep = (P[:, 2] < REGION_Z) & (P[:, 1] > REGION_Y) & (lab < 5)
        else:
            keep = np.ones(len(P), bool)
        remap = -np.ones(len(P), int)
        remap[keep] = np.arange(keep.sum())
        tk = keep[T].all(1)
        out[key + "_P"] = P[keep]
        out[key + "_L"] = lab[keep]
        out[key + "_T"] = remap[T[tk]]
        print(key, "verts", int(keep.sum()), "tris", int(tk.sum()), {LABELS[i]: int((lab[keep] == i).sum()) for i in range(6) if (lab[keep] == i).any()})
    np.savez_compressed(dst, labels=np.array(LABELS), **out)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
