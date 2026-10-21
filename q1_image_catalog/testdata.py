"""Deterministic generator for the test images listed in question 1, section 6.

The images are drawn rather than downloaded so the repo is self-contained and
every run produces identical pixels (tests can assert exact behaviour).
Three extra images cover the failure paths:
  blank      - nothing to detect
  faint      - an object too close to the background colour (detector misses it)
  cluttered  - photo-like noisy background the classical detector cannot model
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WHITE = (255, 255, 255)
RED = (210, 25, 30)
ORANGE = (255, 140, 0)
BROWN = (110, 60, 20)
LEAF = (40, 150, 40)
SKIN = (240, 200, 160)
BLUE = (30, 90, 220)
GREEN = (30, 170, 60)
YELLOW = (245, 215, 20)
PURPLE = (128, 30, 140)
NAVY = (50, 50, 120)
BLACK = (20, 20, 20)
PINK = (240, 110, 170)
DARK_HAIR = (70, 40, 20)


def _apple(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int = 110) -> None:
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=RED)
    d.rectangle((cx - 5, cy - r - 30, cx + 5, cy - r + 15), fill=BROWN)
    d.ellipse((cx + 5, cy - r - 28, cx + 55, cy - r - 2), fill=LEAF)


def _orange(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int = 110) -> None:
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ORANGE)
    d.ellipse((cx - 10, cy - r + 4, cx + 10, cy - r + 18), fill=LEAF)


def _boy(d: ImageDraw.ImageDraw, ox: int = 0) -> None:
    d.rectangle((ox + 150, 130, ox + 250, 250), fill=BLUE)             # shirt
    d.rectangle((ox + 120, 135, ox + 150, 230), fill=SKIN)             # arms
    d.rectangle((ox + 250, 135, ox + 280, 230), fill=SKIN)
    d.rectangle((ox + 155, 250, ox + 195, 360), fill=NAVY)             # trousers
    d.rectangle((ox + 205, 250, ox + 245, 360), fill=NAVY)
    d.rectangle((ox + 145, 360, ox + 197, 378), fill=BLACK)            # shoes
    d.rectangle((ox + 203, 360, ox + 255, 378), fill=BLACK)
    d.ellipse((ox + 160, 50, ox + 240, 132), fill=SKIN)                # head
    d.pieslice((ox + 158, 46, ox + 242, 120), 180, 360, fill=DARK_HAIR)  # short hair


def _girl(d: ImageDraw.ImageDraw, ox: int = 0) -> None:
    d.ellipse((ox + 150, 40, ox + 250, 210), fill=DARK_HAIR)           # long hair
    d.polygon([(ox + 200, 130), (ox + 120, 300), (ox + 280, 300)], fill=PINK)  # dress
    d.rectangle((ox + 110, 150, ox + 140, 240), fill=SKIN)             # arms
    d.rectangle((ox + 260, 150, ox + 290, 240), fill=SKIN)
    d.rectangle((ox + 140, 140, ox + 260, 165), fill=PINK)             # shoulders
    d.rectangle((ox + 170, 300, ox + 190, 370), fill=SKIN)             # legs
    d.rectangle((ox + 210, 300, ox + 230, 370), fill=SKIN)
    d.rectangle((ox + 160, 370, ox + 195, 385), fill=RED)              # shoes
    d.rectangle((ox + 205, 370, ox + 240, 385), fill=RED)
    d.ellipse((ox + 162, 55, ox + 238, 135), fill=SKIN)                # face


def _canvas(w: int, h: int, color=WHITE) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), color)
    return img, ImageDraw.Draw(img)


def make_images() -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}

    img, d = _canvas(400, 400); _apple(d, 200, 220); images["apple"] = img
    img, d = _canvas(400, 400); _orange(d, 200, 210); images["orange"] = img
    img, d = _canvas(400, 420); _boy(d); images["boy"] = img
    img, d = _canvas(400, 420); _girl(d); images["girl"] = img

    img, d = _canvas(860, 300)
    d.ellipse((40, 80, 180, 220), fill=RED)                            # circle
    d.rectangle((240, 80, 380, 220), fill=BLUE)                        # square
    d.polygon([(510, 70), (430, 220), (590, 220)], fill=GREEN)         # triangle
    d.rectangle((650, 105, 820, 195), fill=PURPLE)                     # rectangle
    images["shapes"] = img

    img, d = _canvas(720, 200)
    for i, c in enumerate([RED, BLUE, GREEN, YELLOW, PURPLE]):
        d.rectangle((30 + i * 135, 50, 130 + i * 135, 150), fill=c)
    images["colors"] = img

    img, d = _canvas(800, 400); _apple(d, 200, 220); _orange(d, 580, 210); images["apple_orange"] = img

    img, d = _canvas(520, 400)
    d.ellipse((60, 80, 260, 280), fill=RED)                            # bottom layer
    d.rectangle((200, 140, 360, 300), fill=BLUE)                       # overlaps circle
    d.polygon([(400, 90), (300, 260), (480, 260)], fill=GREEN)         # overlaps square
    images["overlapping_shapes"] = img

    img, _ = _canvas(400, 400); images["blank"] = img
    img, d = _canvas(400, 400); d.ellipse((100, 100, 300, 300), fill=(246, 244, 236)); images["faint"] = img

    rng = np.random.default_rng(7)
    noise = rng.integers(60, 200, size=(400, 400, 3), dtype=np.uint8)
    img = Image.fromarray(noise); d = ImageDraw.Draw(img); _apple(d, 200, 220, r=90)
    images["cluttered"] = img
    return images


def write_images(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, img in make_images().items():
        paths[name] = out / f"{name}.png"
        img.save(paths[name])
    return paths


DESCRIPTIONS = {
    "apple": "A single red apple on a white background",
    "orange": "A single orange fruit on a white background",
    "boy": "A cartoon boy standing",
    "girl": "A cartoon girl standing",
    "shapes": "Circle, square, triangle, and rectangle",
    "colors": "Red, blue, green, yellow, and purple color swatches",
    "apple_orange": "An apple beside an orange",
    "overlapping_shapes": "Overlapping colored geometric shapes",
    "blank": "Empty white image - nothing to detect",
    "faint": "A near-white circle on white - expected detector miss",
    "cluttered": "Apple on a noisy, photo-like background - expected detector failure",
}
