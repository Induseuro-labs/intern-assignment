"""Colour naming: map pixels to basic colour terms in CIE Lab space."""
from __future__ import annotations

import cv2
import numpy as np

# Several sRGB prototypes per basic colour term (light / mid / dark shades).
PROTOTYPES: dict[str, list[tuple[int, int, int]]] = {
    "red": [(220, 40, 40), (180, 20, 30), (130, 10, 20), (235, 80, 70), (200, 30, 60)],
    "orange": [(245, 140, 30), (230, 110, 20), (250, 170, 60), (210, 100, 10)],
    "yellow": [(245, 205, 30), (250, 230, 90), (220, 190, 20)],
    "green": [(40, 170, 70), (20, 110, 40), (120, 200, 90), (60, 140, 60)],
    "blue": [(40, 90, 220), (30, 60, 150), (90, 150, 230), (40, 120, 200), (20, 40, 110)],
    "purple": [(140, 60, 180), (100, 40, 130), (170, 110, 210), (120, 80, 200)],
    "pink": [(240, 120, 170), (250, 180, 200), (225, 90, 140), (240, 150, 160)],
    "brown": [(130, 80, 40), (90, 55, 30), (160, 110, 70), (110, 70, 50)],
    "black": [(15, 15, 15), (45, 40, 40)],
    "white": [(250, 250, 250), (232, 232, 232)],
    "gray": [(128, 128, 128), (180, 180, 180), (80, 80, 80)],
    "skin": [(250, 210, 180), (240, 190, 160), (225, 170, 135), (200, 150, 115)],
}
NEUTRALS = {"black", "white", "gray"}

_names: list[str] = []
_protos = []
for _n, _cs in PROTOTYPES.items():
    for _c in _cs:
        _names.append(_n)
        _protos.append(_c)
_PROTO_LAB = cv2.cvtColor(np.array([_protos], np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def name_lab(lab: np.ndarray) -> np.ndarray:
    """Return an index into ``_names`` for every Lab pixel (N x 3)."""
    d = ((lab[:, None, :] - _PROTO_LAB[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def name_rgb(rgb: tuple[int, int, int]) -> str:
    lab = to_lab(np.array([[rgb]], np.uint8)).reshape(1, 3)
    return _names[int(name_lab(lab)[0])]


def color_histogram(rgb: np.ndarray, mask: np.ndarray, max_pixels: int = 40000,
                    exclude: set[str] | None = None) -> list[dict]:
    """Named colour distribution inside ``mask``; ``[{name, hex, fraction}]``."""
    px = rgb[mask.astype(bool)]
    if len(px) == 0:
        return []
    if len(px) > max_pixels:
        px = px[np.random.default_rng(0).choice(len(px), max_pixels, replace=False)]
    idx = name_lab(to_lab(px[None]).reshape(-1, 3))
    labels = np.array(_names)[idx]
    out = []
    for name in dict.fromkeys(labels):
        if exclude and name in exclude:
            continue
        sel = labels == name
        mean = px[sel].mean(0).astype(int)
        out.append({"name": name, "hex": "#%02x%02x%02x" % tuple(mean), "fraction": float(sel.mean())})
    total = sum(c["fraction"] for c in out) or 1.0
    for c in out:
        c["fraction"] = round(c["fraction"] / total, 3)
    out.sort(key=lambda c: -c["fraction"])
    return [c for c in out if c["fraction"] >= 0.02]


def dominant_color(colors: list[dict]) -> str | None:
    """Most frequent chromatic colour; neutrals only if nothing chromatic is significant."""
    chromatic = [c for c in colors if c["name"] not in NEUTRALS | {"skin"}]
    if chromatic and chromatic[0]["fraction"] >= 0.15:
        return chromatic[0]["name"]
    return colors[0]["name"] if colors else None
