"""Create the Question 1 test images.

* Geometric images (shapes, colors, overlapping_shapes, blank, noise) are drawn
  procedurally so their ground truth is exact. Ground truth is written next to
  them as ``ground_truth.json``.
* Photographic / cartoon images (apple, orange, apple_orange, boy, girl) are
  produced once with an image model via OpenRouter and committed to the repo.
  Re-run with ``--regenerate-ai`` to refresh them (needs OPENROUTER_API_KEY).
  Without a key, simple procedural stand-ins are drawn instead so the pipeline
  can still be exercised offline.

Usage:  python scripts/make_test_images.py [--regenerate-ai]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_images"
sys.path.insert(0, str(ROOT))

W, H = 800, 600
WHITE = (255, 255, 255)

PALETTE = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 170, 70),
    "yellow": (245, 205, 30),
    "purple": (140, 60, 180),
    "orange": (245, 140, 30),
}


def _box(x0, y0, x1, y1):
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def make_shapes():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    gt = []
    # circle
    d.ellipse((60, 80, 260, 280), fill=PALETTE["red"])
    gt.append({"name": "circle", "color": "red", "bbox": _box(60, 80, 261, 281)})
    # square
    d.rectangle((460, 80, 660, 280), fill=PALETTE["blue"])
    gt.append({"name": "square", "color": "blue", "bbox": _box(460, 80, 661, 281)})
    # triangle
    d.polygon([(160, 340), (270, 540), (50, 540)], fill=PALETTE["green"])
    gt.append({"name": "triangle", "color": "green", "bbox": _box(50, 340, 271, 541)})
    # rectangle
    d.rectangle((400, 380, 740, 510), fill=PALETTE["yellow"])
    gt.append({"name": "rectangle", "color": "yellow", "bbox": _box(400, 380, 741, 511)})
    return img, gt


def make_colors():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    gt = []
    names = ["red", "blue", "green", "yellow", "purple"]
    sw, gap = 120, 30
    x = (W - (5 * sw + 4 * gap)) // 2
    for n in names:
        d.rectangle((x, 200, x + sw - 1, 400 - 1), fill=PALETTE[n])
        gt.append({"name": f"{n} swatch", "color": n, "bbox": _box(x, 200, x + sw, 400)})
        x += sw + gap
    return img, gt


def make_overlapping_shapes():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    gt = []
    # Painter's order: square, circle (overlaps square), triangle (overlaps circle)
    d.rectangle((120, 120, 380, 380), fill=PALETTE["blue"])
    gt.append({"name": "square", "color": "blue", "bbox": _box(120, 120, 381, 381)})
    d.ellipse((290, 200, 530, 440), fill=PALETTE["red"])
    gt.append({"name": "circle", "color": "red", "bbox": _box(290, 200, 531, 441)})
    d.polygon([(560, 100), (700, 360), (440, 360)], fill=PALETTE["green"])
    gt.append({"name": "triangle", "color": "green", "bbox": _box(440, 100, 701, 361)})
    return img, gt


def make_blank():
    return Image.new("RGB", (W, H), WHITE), []


def make_noise():
    rng = np.random.default_rng(7)
    arr = rng.normal(128, 40, (H, W, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1.5))
    return img, []


def make_camouflage():
    """A low-contrast square on a textured background: built to be missed by detection."""
    rng = np.random.default_rng(11)
    arr = rng.normal(120, 35, (H, W, 3))
    arr[220:380, 320:480] += 6  # barely brighter square patch, same texture
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    return img, [{"name": "square", "color": "gray", "bbox": _box(320, 220, 480, 380)}]


def make_truncated():
    """A PNG cut off mid-stream: the header parses, decoding fails."""
    buf = io.BytesIO()
    make_shapes()[0].save(buf, "PNG")
    (OUT / "truncated.png").write_bytes(buf.getvalue()[:1200])


# ---------------------------------------------------------------- AI images
AI_PROMPTS = {
    "apple": "A professional product photograph of a single red apple on a plain pure white "
    "background. The whole apple is visible and centred, with plenty of white space around it. "
    "Photorealistic, soft studio light. No other objects, no text.",
    "orange": "A professional product photograph of a single whole orange (the citrus fruit) on a "
    "plain pure white background. The whole fruit is visible and centred with plenty of white space "
    "around it. Photorealistic, soft studio light. No other objects, no text.",
    "apple_orange": "A professional product photograph of one red apple beside one whole orange "
    "(citrus fruit), side by side and not touching, on a plain pure white background. Both fruits "
    "fully visible. Photorealistic, soft studio light. No other objects, no text.",
    "boy": "A flat 2D cartoon illustration of a young boy standing, full body from head to shoes, "
    "facing forward, wearing a blue t-shirt and green shorts. Plain pure white background, no "
    "ground, no shadow, no other objects, no text. Bold clean outlines, flat colours.",
    "girl": "A flat 2D cartoon illustration of a young girl standing, full body from head to shoes, "
    "facing forward, wearing a pink dress and purple shoes, with long brown hair. Plain pure white "
    "background, no ground, no shadow, no other objects, no text. Bold clean outlines, flat colours.",
}


def generate_ai_image(prompt: str) -> Image.Image:
    import httpx

    from imgcat.llm import openrouter_key

    key = openrouter_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("IMAGE_MODEL", "google/gemini-3.1-flash-image"),
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=180,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    url = msg["images"][0]["image_url"]["url"]
    data = base64.b64decode(url.split(",", 1)[1])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1024, 1024))
    return img


def fallback_fruit(color, stem=True):
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    cx, cy, r = W // 2, H // 2 + 20, 160
    for i in range(r, 0, -2):  # radial shading
        t = i / r
        c = tuple(int(v * (0.55 + 0.45 * (1 - t) ** 0.5)) for v in color)
        d.ellipse((cx - i, cy - i, cx + i, cy + i), fill=c)
    if stem:
        d.line((cx, cy - r + 10, cx + 15, cy - r - 40), fill=(90, 60, 30), width=10)
    return img


def fallback_person(shirt, bottom, hair, dress=False):
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    cx = W // 2
    skin = (250, 205, 170)
    d.ellipse((cx - 50, 60, cx + 50, 160), fill=skin, outline="black", width=3)
    d.chord((cx - 52, 52, cx + 52, 140), 180, 360, fill=hair)
    if dress:
        d.polygon([(cx - 45, 165), (cx + 45, 165), (cx + 90, 400), (cx - 90, 400)], fill=shirt, outline="black")
    else:
        d.rectangle((cx - 50, 165, cx + 50, 320), fill=shirt, outline="black", width=3)
        d.rectangle((cx - 50, 320, cx + 50, 400), fill=bottom, outline="black", width=3)
    d.rectangle((cx - 40, 400, cx - 12, 520), fill=skin, outline="black", width=3)
    d.rectangle((cx + 12, 400, cx + 40, 520), fill=skin, outline="black", width=3)
    d.rectangle((cx - 90, 170, cx - 55, 330), fill=skin, outline="black", width=3)
    d.rectangle((cx + 55, 170, cx + 90, 330), fill=skin, outline="black", width=3)
    return img


FALLBACKS = {
    "apple": lambda: fallback_fruit((210, 30, 35)),
    "orange": lambda: fallback_fruit((245, 145, 25), stem=False),
    "boy": lambda: fallback_person((40, 90, 220), (40, 160, 70), (60, 40, 20)),
    "girl": lambda: fallback_person((240, 120, 170), (240, 120, 170), (110, 70, 30), dress=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate-ai", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    gt_path = OUT / "ground_truth.json"
    truth = json.loads(gt_path.read_text()) if gt_path.exists() else {}

    for name, fn in {
        "shapes": make_shapes,
        "colors": make_colors,
        "overlapping_shapes": make_overlapping_shapes,
        "blank": make_blank,
        "noise": make_noise,
        "camouflage": make_camouflage,
    }.items():
        img, gt = fn()
        img.save(OUT / f"{name}.png")
        truth[name] = gt
        print("drew", name)

    make_truncated()
    print("wrote truncated")

    for name, prompt in AI_PROMPTS.items():
        path = OUT / f"{name}.png"
        if path.exists() and not args.regenerate_ai:
            continue
        try:
            img = generate_ai_image(prompt)
            print("generated", name, img.size)
        except Exception as exc:  # offline fallback
            if name == "apple_orange":
                a, o = FALLBACKS["apple"](), FALLBACKS["orange"]()
                img = Image.new("RGB", (W * 2, H), WHITE)
                img.paste(a, (0, 0))
                img.paste(o, (W, 0))
            else:
                img = FALLBACKS[name]()
            print(f"fallback {name}: {exc}")
        img.save(path)

    gt_path.write_text(json.dumps(truth, indent=2))


if __name__ == "__main__":
    main()
