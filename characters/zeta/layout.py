"""Proportions and joint positions (meters). Z up, faces -Y, character's left = +X. Height ~155 cm."""
import math

from mathutils import Matrix, Vector as V

SIDES = {"L": 1.0, "R": -1.0}

SOLE = 0.035
HEAD_C = V((0, 0.0, 1.455))
HEAD_R = (0.088, 0.094, 0.104)  # x, y, z radii of the cranium ellipsoid
PELVIS_Z, WAIST_Z, BUST_Z, SHOULDER_Z, NECK_BASE_Z, CHIN_Z, HEAD_TOP = 0.90, 1.00, 1.13, 1.265, 1.30, 1.355, 1.565

# torso cross sections: z, half-width (x), half-depth (y), center y shift
TORSO = [
    (0.80, 0.120, 0.086, 0.0),
    (0.86, 0.140, 0.092, 0.0),
    (0.915, 0.144, 0.094, 0.002),
    (0.96, 0.112, 0.078, 0.0),
    (1.00, 0.094, 0.068, 0.0),
    (1.06, 0.098, 0.072, -0.004),
    (1.12, 0.106, 0.084, -0.010),
    (1.17, 0.110, 0.082, -0.006),
    (1.22, 0.116, 0.074, 0.0),
    (1.26, 0.128, 0.066, 0.004),
    (1.285, 0.088, 0.058, 0.006),
    (1.30, 0.054, 0.053, 0.008),
    (1.345, 0.046, 0.049, 0.012),
    (1.395, 0.045, 0.048, 0.014),
]

def torso_at(z):
    """Interpolated (half_width, half_depth, y_shift) of the torso at height z."""
    if z <= TORSO[0][0]:
        return TORSO[0][1:]
    for a, b in zip(TORSO, TORSO[1:]):
        if a[0] <= z <= b[0]:
            t = (z - a[0]) / (b[0] - a[0])
            return tuple(a[i] + (b[i] - a[i]) * t for i in (1, 2, 3))
    return TORSO[-1][1:]


def leg_pts(s):
    return {
        "hip": V((s * 0.085, 0.005, 0.845)),
        "knee": V((s * 0.082, -0.012, 0.47)),
        "ankle": V((s * 0.078, 0.02, 0.125)),
        "ball": V((s * 0.078, -0.075, 0.062)),
        "toe": V((s * 0.078, -0.135, 0.050)),
    }


def arm_pts(s):
    return {
        "clav": V((s * 0.03, 0.0, 1.272)),
        "shoulder": V((s * 0.165, 0.0, 1.262)),
        "elbow": V((s * 0.235, 0.018, 1.015)),
        "wrist": V((s * 0.283, 0.004, 0.785)),
    }


def hand_geometry(s, curl=14.0):
    """Joint positions of a relaxed hand hanging along the forearm. Palm faces the body."""
    a = arm_pts(s)
    axis = (a["wrist"] - a["elbow"]).normalized()
    med = V((-s, 0, 0))  # toward the body
    wrist = a["wrist"]
    knuckle = wrist + axis * 0.088
    out = {"wrist": wrist, "knuckle": knuckle, "axis": axis, "palm": [wrist + axis * (0.088 * t) for t in (0, 0.35, 0.7, 1.0)]}
    # fingers: y offset (front negative), segment lengths
    spec = {
        "index": (-0.0215, (0.034, 0.022, 0.019)),
        "middle": (-0.0075, (0.038, 0.025, 0.020)),
        "ring": (0.0065, (0.034, 0.023, 0.019)),
        "pinky": (0.019, (0.027, 0.017, 0.016)),
    }
    fingers = {}
    for name, (yoff, lens) in spec.items():
        p = knuckle + V((0, yoff, 0))
        d = axis.copy()
        pts = [p]
        for k, ln in enumerate(lens):
            ang = math.radians(curl * (k + 1) * 0.8)
            R = Matrix.Rotation(ang * (1 if s > 0 else -1), 3, "Y")
            d = (R @ axis).normalized() if k == 0 else (R @ d).normalized()
            p = p + d * ln
            pts.append(p.copy())
        fingers[name] = pts
    # thumb: starts near the wrist on the front side, points down/front/medial
    tp = wrist + axis * 0.026 + V((0, -0.026, 0)) + med * 0.004
    td = (axis * 0.8 + V((0, -0.55, 0)) + med * 0.25).normalized()
    pts = [tp]
    for ln in (0.034, 0.028, 0.024):
        tp = tp + td * ln
        td = (td + axis * 0.18 + med * 0.08).normalized()
        pts.append(tp.copy())
    fingers["thumb"] = pts
    out["fingers"] = fingers
    return out
