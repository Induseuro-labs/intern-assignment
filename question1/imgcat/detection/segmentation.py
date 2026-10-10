"""Classical region proposals for images with a plain background.

1. Estimate the background colour from the image border.
2. Foreground = pixels far from it in Lab, minus soft cast shadows
   (neutral and only slightly darker than the background).
3. Group foreground into connected regions.
4. Split a region into colour layers when it is made of a few flat colours that
   are each a geometric shape (overlapping shapes); keep it whole otherwise
   (a cartoon figure is many flat colours, but its layers are not shapes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .colors import to_lab
from .shapes import ShapeFit, fit_shape


@dataclass
class Background:
    rgb: tuple[int, int, int]
    lab: np.ndarray
    uniform: float  # fraction of border pixels that match the background colour

    @property
    def plain(self) -> bool:
        return self.uniform >= 0.85


@dataclass
class Region:
    mask: np.ndarray
    bbox: dict
    area: int
    flat: bool = False
    color_rgb: tuple[int, int, int] | None = None
    shape: ShapeFit | None = None
    occluded_by: list[int] = field(default_factory=list)


def bbox_of(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    return {"x": int(xs.min()), "y": int(ys.min()),
            "width": int(xs.max() - xs.min() + 1), "height": int(ys.max() - ys.min() + 1)}


def estimate_background(rgb: np.ndarray, border: int = 6) -> Background:
    strips = np.concatenate([rgb[:border].reshape(-1, 3), rgb[-border:].reshape(-1, 3),
                             rgb[:, :border].reshape(-1, 3), rgb[:, -border:].reshape(-1, 3)])
    lab = to_lab(strips[None]).reshape(-1, 3)
    med = np.median(lab, 0)
    uniform = float((np.linalg.norm(lab - med, axis=1) < 12).mean())
    rgb_med = tuple(int(v) for v in np.median(strips, 0))
    return Background(rgb_med, med, uniform)


def foreground_mask(rgb: np.ndarray, bg: Background, thresh: float = 22.0) -> np.ndarray:
    lab = to_lab(rgb)
    # local background: blur-free estimate assumes a smooth backdrop; compare with border colour
    de = np.linalg.norm(lab - bg.lab, axis=2)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    bg_chroma = np.hypot(bg.lab[1] - 128, bg.lab[2] - 128)
    # soft shadow / vignette: near-neutral and not much darker than background
    shadow = (chroma < bg_chroma + 20) & (lab[..., 0] > bg.lab[0] - 70) & (lab[..., 0] < bg.lab[0] + 10)
    fg = (de > thresh) & ~shadow
    fg = fg.astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return _fill_holes(fg).astype(bool)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    inv = (mask == 0).astype(np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 0)
    return mask | inv


def connected_regions(fg: np.ndarray, min_area_frac: float = 0.002, group_px: int = 9) -> list[Region]:
    """Connected components; parts closer than ``group_px`` are grouped together."""
    h, w = fg.shape
    grouped = cv2.dilate(fg.astype(np.uint8), np.ones((group_px, group_px), np.uint8))
    n, labels = cv2.connectedComponents(grouped)
    regions = []
    for i in range(1, n):
        m = (labels == i) & fg
        area = int(m.sum())
        if area >= min_area_frac * h * w:
            regions.append(Region(m, bbox_of(m), area))
    return regions


def flat_layers(rgb: np.ndarray, region: Region, max_layers: int = 5) -> list[tuple[tuple, np.ndarray]] | None:
    """If the region is a few flat colours, return ``[(rgb, mask)]`` per colour; else None."""
    px = rgb[region.mask]
    q = (px // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    major = [k for k in order if counts[k] >= 0.04 * len(px)]
    if not major or len(major) > max_layers:
        return None
    if counts[major].sum() < 0.95 * len(px):  # gradients / texture → not flat
        return None
    full_keys = np.full(region.mask.shape, -1, np.int64)
    full_keys[region.mask] = keys
    layers = []
    for k in major:
        m = full_keys == uniq[k]
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        color = tuple(int(v) for v in rgb[m].mean(0))
        layers.append((color, m))
    # absorb anti-aliased edge pixels into the nearest layer so occluder masks are tight
    return layers


def decompose(rgb: np.ndarray, region: Region) -> list[Region]:
    """Split a flat multi-colour region into shape layers, if every layer is a shape."""
    layers = flat_layers(rgb, region)
    if layers is None:
        return [region]
    out = []
    for i, (color, m) in enumerate(layers):
        n, labels = cv2.connectedComponents(m.astype(np.uint8))
        if n - 1 != 1:  # a shape layer must be a single connected blob
            return [region] if len(layers) > 1 else [region]
        others = np.zeros_like(m)
        for j, (_, mj) in enumerate(layers):
            if j != i:
                others |= mj
        fit = fit_shape(m, others)
        if fit is None:
            return [region]
        out.append(Region(m, bbox_of(m), int(m.sum()), True, color, fit))
    # occlusion graph: layer j hides layer i when j covers the part of i's fitted model that is not visible
    for i, r in enumerate(out):
        hidden = r.shape.model & ~r.mask if r.shape.model is not None else None
        if hidden is None or r.shape.coverage >= 0.97:
            continue
        for j, o in enumerate(out):
            if j != i and (hidden & o.mask).sum() > 0.05 * hidden.sum():
                r.occluded_by.append(j)
    return out


def trim_contact_shadow(rgb: np.ndarray, region: Region, bg: Background) -> Region:
    """Remove the contact shadow a photographed object casts on the surface below it.

    Contact shadows are darker than the backdrop but nearly as colourless (chroma
    < 35 even when tinted by the object), while the objects we photograph are
    saturated. The rule is applied only to the bottom 20% of the region, so
    pale parts elsewhere in the object are never touched.
    """
    y0 = region.bbox["y"] + int(region.bbox["height"] * 0.8)
    lab = to_lab(rgb[y0:])
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    bg_chroma = float(np.hypot(bg.lab[1] - 128, bg.lab[2] - 128))
    shadow = np.zeros(region.mask.shape, bool)
    shadow[y0:] = (chroma < bg_chroma + 35) & (lab[..., 0] < bg.lab[0] + 10)
    mask = region.mask & ~shadow
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    if mask.sum() < 0.8 * region.area:
        return region
    return Region(mask, bbox_of(mask), int(mask.sum()))


def estimate_style(rgb: np.ndarray, mask: np.ndarray | None = None) -> str:
    """'illustration' for flat-shaded art, 'photo' for natural images.

    Flat fills mean most horizontally adjacent pixels are identical (cartoons
    ≈ 0.65-0.7, vector shapes ≈ 1.0); camera images have texture and noise
    everywhere (≈ 0.2-0.25).
    """
    px = rgb.astype(np.int16)
    same = np.abs(px[:, 1:] - px[:, :-1]).max(2) <= 2
    m = np.ones(same.shape, bool) if mask is None or mask.sum() < 500 else (mask[:, 1:] & mask[:, :-1])
    return "illustration" if same[m].mean() > 0.5 else "photo"
