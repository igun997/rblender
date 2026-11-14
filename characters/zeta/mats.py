"""Procedural textures (numpy -> packed images, so they survive glTF export) and the material set."""
import numpy as np

from .common import image_from_array, material

SKIN = (0.96, 0.80, 0.72)
HAIR_MID = (0.70, 0.71, 0.77)
WHITE = (0.92, 0.92, 0.95)
BLACK = (0.035, 0.035, 0.045)
OLIVE = (0.30, 0.27, 0.16)
BLUE = (0.22, 0.34, 0.85)


def _grid(h, w):
    y, x = np.mgrid[0:h, 0:w]
    return x / (w - 1), y / (h - 1)  # u, v (v=0 bottom)


def _ell(u, v, cx, cy, rx, ry):
    return ((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2


def eye_texture(size=256):
    u, v = _grid(size, size)
    img = np.zeros((size, size, 3))
    sclera = np.array([0.97, 0.98, 1.0])
    img[:] = sclera
    img *= (1.0 - 0.22 * np.clip((v - 0.78) / 0.22, 0, 1) ** 1.5)[..., None]  # lid shadow on top
    d_iris = _ell(u, v, 0.5, 0.45, 0.40, 0.45)
    t = np.clip((v - 0.0) / 0.9, 0, 1)  # 0 bottom, 1 top
    top, bot = np.array([0.20, 0.30, 0.58]), np.array([0.62, 0.78, 0.98])
    iris = bot[None, None, :] * (1 - t[..., None]) + top[None, None, :] * t[..., None]
    rim = np.clip((d_iris - 0.62) / 0.38, 0, 1)[..., None]
    iris = iris * (1 - rim) + np.array([0.10, 0.15, 0.34]) * rim
    m = np.clip((1.0 - d_iris) * 40, 0, 1)[..., None]
    img = img * (1 - m) + iris * m
    d_pup = _ell(u, v, 0.5, 0.47, 0.13, 0.19)
    mp = np.clip((1.0 - d_pup) * 40, 0, 1)[..., None]
    img = img * (1 - mp) + np.array([0.03, 0.05, 0.12]) * mp
    for cx, cy, r in ((0.36, 0.66, 0.085), (0.63, 0.30, 0.045), (0.58, 0.72, 0.022)):
        d = _ell(u, v, cx, cy, r, r * 1.05)
        mh = np.clip((1.0 - d) * 30, 0, 1)[..., None]
        img = img * (1 - mh) + np.array([1.0, 1.0, 1.0]) * mh
    return image_from_array("zeta_eye", img)


def hair_texture():
    h, w = 256, 16
    u, v = _grid(h, w)
    stops = [(0.0, (0.30, 0.31, 0.37)), (0.25, (0.46, 0.47, 0.53)), (0.7, (0.58, 0.59, 0.66)), (1.0, (0.74, 0.76, 0.84))]
    xs = [s[0] for s in stops]
    col = np.stack([np.interp(v, xs, [s[1][c] for s in stops]) for c in range(3)], axis=-1)
    edge = 0.86 + 0.14 * np.sin(np.pi * u)
    return image_from_array("zeta_hair", col * edge[..., None])


def lattice_texture(name, w=256, h=512, top=(0.07, 0.07, 0.09), bottom=(0.46, 0.48, 0.54), spacing=64,
                    line=0.10, border=None):
    u, v = _grid(h, w)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    t = np.clip(v, 0, 1)[..., None]  # v=1 top of panel
    base = np.array(bottom) * (1 - t) + np.array(top) * t
    on = (((X + Y) % spacing) < 3) | (((X - Y) % spacing) < 3)
    img = base + on[..., None] * line
    if border is not None:
        bw = max(w // 40, 3)
        edge = (X < bw)[..., None]
        img = np.where(edge, np.array(border), img)
    return image_from_array(name, img)


def fishnet_texture(size=256, spacing=36):
    u, v = _grid(size, size)
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    img = np.ones((size, size, 3)) * np.array([0.93, 0.93, 0.96])
    on = (((X + Y) % spacing) < 4) | (((X - Y) % spacing) < 4)
    img = np.where(on[..., None], np.array([0.55, 0.56, 0.62]), img)
    return image_from_array("zeta_fishnet", img)


def diamond_chain_texture(w=64, h=512):
    """Vertical chain of alternating black/white diamonds on a mid field (front skirt panel)."""
    u, v = _grid(h, w)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    img = np.ones((h, w, 3)) * np.array([0.12, 0.12, 0.15])
    period = 64
    cy = (Y % period) - period / 2
    d = np.abs(X - w / 2) / (w * 0.42) + np.abs(cy) / (period * 0.45)
    ring = (d < 1.0) & (d > 0.80)
    fill = d <= 0.80
    idx = (Y // period) % 2
    img = np.where(ring[..., None], np.array([0.95, 0.95, 0.98]), img)
    img = np.where((fill & (idx == 0))[..., None], np.array([0.95, 0.95, 0.98]), img)
    return image_from_array("zeta_diamonds", img)


def tablet_texture():
    w, h = 256, 340
    u, v = _grid(h, w)
    img = np.ones((h, w, 3)) * np.array([0.24, 0.36, 0.88])
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    inner = (X > 18) & (X < w - 18) & (Y > 24) & (Y < h - 24)
    img = np.where(inner[..., None], np.array([0.80, 0.86, 1.0]), img)
    for k in range(7):  # fake text lines
        y0 = 60 + k * 36
        ln = (X > 40) & (X < 40 + (w - 100) * (0.45 + 0.5 * ((k * 37) % 10) / 10)) & (Y > y0) & (Y < y0 + 8)
        img = np.where(ln[..., None], np.array([0.25, 0.30, 0.55]), img)
    return image_from_array("zeta_tablet", img)


def build():
    M = {}
    M["skin"] = material("Zeta_Skin", SKIN, rough=0.6)
    M["eye"] = material("Zeta_Eye", (1, 1, 1), rough=0.25, tex=eye_texture())
    M["hair"] = material("Zeta_Hair", HAIR_MID, rough=0.45, tex=hair_texture())
    M["hair_cap"] = material("Zeta_HairCap", (0.40, 0.41, 0.47), rough=0.5)
    M["lash"] = material("Zeta_Lash", (0.05, 0.05, 0.08), rough=1.0, spec=0.0)
    M["brow"] = material("Zeta_Brow", (0.40, 0.41, 0.48), rough=0.7)
    M["mouth"] = material("Zeta_Mouth", (0.78, 0.35, 0.40), rough=0.5)
    M["mouth_in"] = material("Zeta_MouthInside", (0.30, 0.08, 0.12), rough=0.5)
    M["nose"] = material("Zeta_Nose", (0.90, 0.62, 0.58), rough=0.6)
    M["white"] = material("Zeta_DressWhite", WHITE, rough=0.7)
    M["black"] = material("Zeta_Black", BLACK, rough=0.5)
    M["leather"] = material("Zeta_Leather", (0.06, 0.06, 0.08), rough=0.35, spec=0.6)
    M["olive"] = material("Zeta_Olive", OLIVE, rough=0.75)
    M["blue"] = material("Zeta_Blue", BLUE, rough=0.4)
    M["gem"] = material("Zeta_Gem", (0.45, 0.55, 1.0), rough=0.15, emission=(0.2, 0.3, 0.8))
    M["gold"] = material("Zeta_Gold", (0.80, 0.62, 0.25), rough=0.3, metal=0.8)
    M["cape"] = material("Zeta_Cape", (1, 1, 1), rough=0.65,
                         tex=lattice_texture("zeta_cape", border=(0.25, 0.38, 0.9)))
    M["sock"] = material("Zeta_Sock", (1, 1, 1), rough=0.8, tex=fishnet_texture())
    M["stocking"] = material("Zeta_Stocking", (0.05, 0.045, 0.05), rough=0.55)
    M["diamonds"] = material("Zeta_DiamondPanel", (1, 1, 1), rough=0.6, tex=diamond_chain_texture())
    M["tablet"] = material("Zeta_Tablet", (1, 1, 1), rough=0.4, tex=tablet_texture())
    M["silver"] = material("Zeta_Silver", (0.75, 0.77, 0.82), rough=0.25, metal=0.9)
    return M
