"""
Script to programmatically generate all required test images for Question 1.
Images generated:
1. apple.png - Single red apple with stem/leaf on white background
2. orange.png - Single orange fruit on white background
3. boy.png - Cartoon boy character standing with blue shirt and dark shorts
4. girl.png - Cartoon girl character standing with pink dress and blonde hair
5. shapes.png - Circle, square, triangle, and rectangle
6. colors.png - Red, blue, green, yellow, and purple color swatches
7. apple_orange.png - Apple beside an orange
8. overlapping_shapes.png - Overlapping colored geometric shapes
9. empty_canvas.png - White canvas with subtle gradient (no detectable objects)
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def create_apple():
    """Generates a high-quality red apple image with stem and green leaf on white background."""
    img = Image.new("RGBA", (500, 500), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Apple body (overlapping circles for natural apple heart shape)
    apple_color = (220, 20, 60, 255)  # Crimson red
    draw.ellipse([140, 150, 310, 390], fill=apple_color)
    draw.ellipse([190, 150, 360, 390], fill=apple_color)
    draw.ellipse([160, 170, 340, 400], fill=apple_color)
    # Highlight
    draw.ellipse([170, 180, 210, 240], fill=(255, 100, 120, 180))
    
    # Stem
    draw.line([(250, 160), (255, 110)], fill=(101, 67, 33, 255), width=8)
    # Green leaf
    leaf_color = (34, 139, 34, 255)
    draw.polygon([(255, 120), (290, 100), (320, 115), (280, 130)], fill=leaf_color)
    
    rgb_img = Image.new("RGB", (500, 500), (255, 255, 255))
    rgb_img.paste(img, (0, 0), img)
    rgb_img.save(os.path.join(DATA_DIR, "apple.png"))
    print("Created apple.png")

def create_orange():
    """Generates a single orange fruit on white background."""
    img = Image.new("RGBA", (500, 500), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    orange_color = (255, 140, 0, 255)  # Dark orange
    draw.ellipse([130, 130, 370, 370], fill=orange_color)
    # Highlight & texture
    draw.ellipse([160, 150, 220, 210], fill=(255, 175, 50, 200))
    
    # Stem point & small green leaf
    draw.ellipse([245, 130, 255, 140], fill=(90, 60, 20, 255))
    draw.polygon([(250, 135), (280, 110), (300, 125), (265, 135)], fill=(34, 139, 34, 255))
    
    rgb_img = Image.new("RGB", (500, 500), (255, 255, 255))
    rgb_img.paste(img, (0, 0), img)
    rgb_img.save(os.path.join(DATA_DIR, "orange.png"))
    print("Created orange.png")

def create_boy():
    """Generates cartoon boy standing with blue shirt and dark navy pants."""
    img = Image.new("RGBA", (500, 600), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    skin = (255, 218, 185, 255)
    hair = (101, 67, 33, 255)
    shirt_blue = (30, 144, 255, 255)
    pants_navy = (25, 25, 112, 255)
    shoes_black = (40, 40, 40, 255)
    
    # Head & ears
    draw.ellipse([210, 80, 290, 170], fill=skin)
    draw.ellipse([200, 115, 215, 135], fill=skin)
    draw.ellipse([285, 115, 300, 135], fill=skin)
    # Hair
    draw.chord([205, 70, 295, 140], start=180, end=360, fill=hair)
    # Eyes & smile
    draw.ellipse([230, 115, 240, 125], fill=(30, 30, 30, 255))
    draw.ellipse([260, 115, 270, 125], fill=(30, 30, 30, 255))
    draw.arc([240, 130, 260, 145], start=0, end=180, fill=(180, 50, 50, 255), width=2)
    
    # Neck
    draw.rectangle([240, 165, 260, 185], fill=skin)
    
    # Blue Shirt (Torso)
    draw.rectangle([210, 185, 290, 320], fill=shirt_blue)
    # Arms
    draw.rectangle([180, 195, 210, 300], fill=shirt_blue)
    draw.rectangle([290, 195, 320, 300], fill=shirt_blue)
    # Hands
    draw.ellipse([180, 295, 210, 325], fill=skin)
    draw.ellipse([290, 295, 320, 325], fill=skin)
    
    # Pants (Legs)
    draw.rectangle([215, 320, 245, 470], fill=pants_navy)
    draw.rectangle([255, 320, 285, 470], fill=pants_navy)
    
    # Shoes
    draw.ellipse([210, 465, 245, 495], fill=shoes_black)
    draw.ellipse([255, 465, 290, 495], fill=shoes_black)
    
    rgb_img = Image.new("RGB", (500, 600), (255, 255, 255))
    rgb_img.paste(img, (0, 0), img)
    rgb_img.save(os.path.join(DATA_DIR, "boy.png"))
    print("Created boy.png")

def create_girl():
    """Generates cartoon girl standing with pink/magenta dress and blonde hair."""
    img = Image.new("RGBA", (500, 600), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    skin = (255, 224, 189, 255)
    hair_blonde = (238, 201, 0, 255)
    dress_pink = (255, 105, 180, 255)
    shoes_red = (220, 20, 60, 255)
    
    # Long Hair background
    draw.ellipse([195, 75, 305, 230], fill=hair_blonde)
    
    # Head & ears
    draw.ellipse([210, 80, 290, 170], fill=skin)
    # Hair bangs & ponytails
    draw.chord([205, 70, 295, 130], start=180, end=360, fill=hair_blonde)
    draw.ellipse([175, 95, 215, 160], fill=hair_blonde)
    draw.ellipse([285, 95, 325, 160], fill=hair_blonde)
    
    # Eyes & smile
    draw.ellipse([230, 115, 240, 125], fill=(30, 30, 30, 255))
    draw.ellipse([260, 115, 270, 125], fill=(30, 30, 30, 255))
    draw.arc([240, 130, 260, 145], start=0, end=180, fill=(200, 40, 40, 255), width=2)
    
    # Neck
    draw.rectangle([240, 165, 260, 185], fill=skin)
    
    # Pink Dress (A-line shape)
    draw.polygon([(230, 185), (270, 185), (315, 350), (185, 350)], fill=dress_pink)
    
    # Arms
    draw.rectangle([170, 195, 195, 290], fill=skin)
    draw.rectangle([305, 195, 330, 290], fill=skin)
    
    # Legs
    draw.rectangle([220, 350, 240, 470], fill=skin)
    draw.rectangle([260, 350, 280, 470], fill=skin)
    
    # Shoes
    draw.ellipse([215, 465, 245, 490], fill=shoes_red)
    draw.ellipse([255, 465, 285, 490], fill=shoes_red)
    
    rgb_img = Image.new("RGB", (500, 600), (255, 255, 255))
    rgb_img.paste(img, (0, 0), img)
    rgb_img.save(os.path.join(DATA_DIR, "girl.png"))
    print("Created girl.png")

def create_shapes():
    """Generates Circle, Square, Triangle, and Rectangle."""
    img = Image.new("RGB", (700, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 1. Circle (Blue)
    draw.ellipse([60, 80, 200, 220], fill=(30, 144, 255))  # Blue circle
    
    # 2. Square (Red)
    draw.rectangle([260, 80, 400, 220], fill=(220, 20, 60)) # Red square
    
    # 3. Triangle (Green)
    draw.polygon([(530, 80), (460, 220), (600, 220)], fill=(34, 139, 34)) # Green triangle
    
    # 4. Rectangle (Yellow / Amber)
    draw.rectangle([150, 300, 550, 420], fill=(255, 190, 0)) # Yellow rectangle
    
    img.save(os.path.join(DATA_DIR, "shapes.png"))
    print("Created shapes.png")

def create_colors():
    """Generates 5 distinct color swatches: Red, Blue, Green, Yellow, and Purple."""
    img = Image.new("RGB", (800, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    swatches = [
        ("red", (220, 20, 60), (40, 100, 160, 260)),
        ("blue", (30, 144, 255), (190, 100, 310, 260)),
        ("green", (34, 139, 34), (340, 100, 460, 260)),
        ("yellow", (255, 215, 0), (490, 100, 610, 260)),
        ("purple", (128, 0, 128), (640, 100, 760, 260)),
    ]
    
    for name, rgb, bbox in swatches:
        draw.rectangle(bbox, fill=rgb, outline=(50, 50, 50), width=2)
    
    img.save(os.path.join(DATA_DIR, "colors.png"))
    print("Created colors.png")

def create_apple_orange():
    """Generates an apple beside an orange on white background."""
    img = Image.new("RGB", (700, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Apple on the left
    apple_color = (220, 20, 60)
    draw.ellipse([90, 160, 260, 390], fill=apple_color)
    draw.ellipse([140, 160, 310, 390], fill=apple_color)
    draw.line([(200, 170), (205, 120)], fill=(101, 67, 33), width=7)
    draw.polygon([(205, 130), (235, 110), (260, 125), (225, 140)], fill=(34, 139, 34))
    
    # Orange on the right
    orange_color = (255, 140, 0)
    draw.ellipse([390, 160, 620, 390], fill=orange_color)
    draw.ellipse([420, 190, 470, 240], fill=(255, 175, 50))
    draw.ellipse([500, 160, 510, 170], fill=(90, 60, 20))
    draw.polygon([(505, 165), (530, 145), (550, 160), (520, 170)], fill=(34, 139, 34))
    
    img.save(os.path.join(DATA_DIR, "apple_orange.png"))
    print("Created apple_orange.png")

def create_overlapping_shapes():
    """Generates overlapping colored geometric shapes: blue circle overlapping yellow square and red triangle."""
    img = Image.new("RGB", (600, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Base: Yellow square
    draw.rectangle([140, 140, 360, 360], fill=(255, 215, 0), outline=(180, 150, 0), width=2)
    
    # Mid-layer: Red triangle overlapping right side of square
    draw.polygon([(300, 100), (200, 320), (450, 320)], fill=(220, 20, 60), outline=(150, 10, 30), width=2)
    
    # Top-layer: Semi-transparent or bold Cyan/Blue circle overlapping both
    draw.ellipse([220, 180, 420, 380], fill=(30, 144, 255), outline=(10, 80, 180), width=2)
    
    img.save(os.path.join(DATA_DIR, "overlapping_shapes.png"))
    print("Created overlapping_shapes.png")

def create_empty_canvas():
    """Generates an image containing no detectable objects (clean white background)."""
    img = Image.new("RGB", (500, 500), (255, 255, 255))
    img.save(os.path.join(DATA_DIR, "empty_canvas.png"))
    print("Created empty_canvas.png")

def generate_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    create_apple()
    create_orange()
    create_boy()
    create_girl()
    create_shapes()
    create_colors()
    create_apple_orange()
    create_overlapping_shapes()
    create_empty_canvas()
    print("All test assets created successfully in", DATA_DIR)

if __name__ == "__main__":
    generate_all()
