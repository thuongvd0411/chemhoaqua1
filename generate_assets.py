from PIL import Image, ImageDraw
import os

ASSETS_DIR = "assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

def create_banana():
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Draw curved yellow shape
    draw.arc((10, 10, 90, 90), start=30, end=150, fill=(255, 255, 0), width=20)
    # Add a stem
    draw.line((15, 20, 25, 10), fill=(0, 100, 0), width=5)
    img.save(os.path.join(ASSETS_DIR, "banana.png"))
    print("Created banana.png")

def create_cherry():
    img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Stems
    draw.line((50, 20, 30, 60), fill=(0, 128, 0), width=3)
    draw.line((50, 20, 70, 60), fill=(0, 128, 0), width=3)
    # Cherries
    draw.ellipse((20, 55, 45, 80), fill=(220, 20, 60), outline=(100, 0, 0))
    draw.ellipse((55, 55, 80, 80), fill=(220, 20, 60), outline=(100, 0, 0))
    # Highlights
    draw.ellipse((25, 60, 30, 65), fill=(255, 255, 255))
    draw.ellipse((60, 60, 65, 65), fill=(255, 255, 255))
    img.save(os.path.join(ASSETS_DIR, "cherry.png"))
    print("Created cherry.png")

if __name__ == "__main__":
    create_banana()
    create_cherry()
