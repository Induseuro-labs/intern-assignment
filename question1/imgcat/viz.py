"""Draw catalog objects onto their image for reports and the UI."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

ORIGIN_COLORS = {"detector": (230, 30, 120), "user": (20, 150, 60), "composition": (30, 90, 230)}


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image_path, objects: list[dict], out_path) -> None:
    """``objects`` are ``CatalogObject.to_dict()`` entries; edited objects get a ✎ marker."""
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 300)
    font = _font(max(12, im.width // 50))
    for o in objects:
        if o.get("deleted"):
            continue
        e = o["effective"]
        b = e["bbox"]
        col = ORIGIN_COLORS["user"] if o["edited_fields"] or o["origin"] == "user" else ORIGIN_COLORS.get(o["origin"], (0, 0, 0))
        d.rectangle((b["x"], b["y"], b["x"] + b["width"] - 1, b["y"] + b["height"] - 1), outline=col, width=lw)
        conf = e.get("confidence")
        label = f"{e.get('display_name') or e['name']} {conf:.2f}" if conf is not None else e["name"]
        if o["edited_fields"] or o["origin"] == "user":
            label += " (user)"
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = b["y"] - th - 6 if b["y"] - th - 6 >= 0 else b["y"] + 2
        d.rectangle((b["x"], ty, b["x"] + tw + 8, ty + th + 6), fill=col)
        d.text((b["x"] + 4, ty + 1), label, fill="white", font=font)
    im.save(out_path)
