"""Q1 test data: deterministic 400x400 catalog images for every brief scenario."""
from PIL import Image, ImageDraw

from detector import PALETTE


def _figure(draw, box_color, dress=None):
    x0, y0 = 150, 40  # cartoon person inside a 400x400 frame
    draw.ellipse([x0 + 20, y0 + 20, x0 + 80, y0 + 80], fill=PALETTE["skin"])     # head
    draw.arc([x0 + 20, y0 + 20, x0 + 80, y0 + 55], 180, 360, fill="black", width=8)  # hair
    draw.rectangle([x0 + 15, y0 + 70, x0 + 85, y0 + 185], fill=box_color)          # torso
    if dress:  # girl: trapezoid dress hides the torso; arms overlap its edge
        draw.polygon([(x0 + 15, y0 + 78), (x0 + 85, y0 + 78),
                      (x0 + 110, y0 + 196), (x0 - 10, y0 + 196)], fill=dress)
        draw.rectangle([x0 + 4, y0 + 95, x0 + 18, y0 + 160], fill=PALETTE["skin"])
        draw.rectangle([x0 + 82, y0 + 95, x0 + 96, y0 + 160], fill=PALETTE["skin"])
        draw.rectangle([x0 + 25, y0 + 195, x0 + 45, y0 + 265], fill=PALETTE["skin"])
        draw.rectangle([x0 + 55, y0 + 195, x0 + 75, y0 + 265], fill=PALETTE["skin"])
        return
    draw.rectangle([x0 + 10, y0 + 95, x0 + 16, y0 + 160], fill=PALETTE["skin"])    # arms
    draw.rectangle([x0 + 84, y0 + 95, x0 + 90, y0 + 160], fill=PALETTE["skin"])
    # boy: pants
    draw.rectangle([x0 + 20, y0 + 185, x0 + 48, y0 + 265], fill=PALETTE["navy"])
    draw.rectangle([x0 + 52, y0 + 185, x0 + 80, y0 + 265], fill=PALETTE["navy"])


def make_test_images():
    """Deterministic 400x400 catalog images for every scenario in the brief."""
    W = lambda: Image.new("RGB", (400, 400), "white")
    imgs = {}
    im = W()
    d = ImageDraw.Draw(im)
    d.ellipse([140, 150, 260, 290], fill=PALETTE["red"])                     # apple
    d.rectangle([196, 118, 204, 152], fill=PALETTE["brown"])
    d.ellipse([206, 122, 246, 148], fill=PALETTE["green"])
    imgs["apple"] = im
    im = W()
    d = ImageDraw.Draw(im)
    d.ellipse([140, 160, 260, 280], fill=PALETTE["orange"])                   # orange
    d.rectangle([196, 132, 204, 162], fill=PALETTE["brown"])
    imgs["orange"] = im
    im = W()
    _figure(ImageDraw.Draw(im), PALETTE["blue"])
    imgs["boy"] = im                                                          # cartoon boy
    im = W()
    _figure(ImageDraw.Draw(im), PALETTE["skin"], dress=PALETTE["pink"])
    imgs["girl"] = im                                                         # cartoon girl
    im = W()
    d = ImageDraw.Draw(im)                                                    # shapes
    d.ellipse([30, 30, 130, 130], fill=PALETTE["red"])
    d.rectangle([160, 30, 260, 130], fill=PALETTE["blue"])
    d.polygon([(340, 35), (295, 125), (385, 125)], fill=PALETTE["green"])
    d.rectangle([30, 180, 200, 260], fill=PALETTE["purple"])
    imgs["shapes"] = im
    im = W()
    d = ImageDraw.Draw(im)                                                    # colors
    for i, c in enumerate(["red", "blue", "green", "yellow", "purple"]):
        d.rectangle([15 + i * 76, 100, 80 + i * 76, 250], fill=PALETTE[c])
    imgs["colors"] = im
    im = W()                                                                  # apple_orange
    im.paste(imgs["apple"].resize((180, 180)), (20, 110))
    im.paste(imgs["orange"].resize((180, 180)), (200, 110))
    imgs["apple_orange"] = im
    im = W()
    d = ImageDraw.Draw(im)                                                    # overlapping
    d.ellipse([80, 80, 240, 240], fill=PALETTE["red"])
    d.rectangle([160, 140, 320, 300], fill=PALETTE["blue"])
    d.polygon([(200, 60), (120, 220), (290, 220)], fill=PALETTE["green"])
    imgs["overlapping_shapes"] = im
    imgs["blank"] = W()                                                       # no objects
    return imgs
