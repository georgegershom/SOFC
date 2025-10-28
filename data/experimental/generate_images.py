import os
import random
from typing import Tuple

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit(
        "Pillow (PIL) is required. Install via: python3 -m pip install --user pillow"
    ) from exc


def ensure_directories(dic_dir: str, sem_dir: str) -> None:
    os.makedirs(dic_dir, exist_ok=True)
    os.makedirs(sem_dir, exist_ok=True)


def generate_speckle_images(output_dir: str, num_images: int = 101, width: int = 640, height: int = 480) -> None:
    ensure_directories(output_dir, output_dir)
    for index in range(num_images):
        image = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(image)
        random.seed(1000 + index)
        for _ in range(5000):
            x_coord = random.randrange(width)
            y_coord = random.randrange(height)
            radius = random.randint(1, 2)
            draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), fill=0)
        image.save(os.path.join(output_dir, f"speckle_{index:03d}.png"))


def generate_sem_placeholder_images(output_dir: str, num_images: int = 5, size: Tuple[int, int] = (512, 512)) -> None:
    ensure_directories(output_dir, output_dir)
    width, height = size
    for index in range(1, num_images + 1):
        image = Image.new("L", (width, height), 200)
        draw = ImageDraw.Draw(image)
        random.seed(2000 + index)

        # Crack-like line segments
        for _ in range(60):
            x1 = random.randrange(width)
            y1 = random.randrange(height)
            x2 = min(max(0, x1 + random.randint(-80, 80)), width - 1)
            y2 = min(max(0, y1 + random.randint(-80, 80)), height - 1)
            draw.line((x1, y1, x2, y2), fill=80, width=random.randint(1, 3))

        # Grainy background texture
        for _ in range(2000):
            x = random.randrange(width)
            y = random.randrange(height)
            value = 180 + random.randint(-15, 15)
            draw.point((x, y), fill=value)

        image.save(os.path.join(output_dir, f"sem_area_{index}.png"))


def main() -> None:
    dic_image_dir = "/workspace/data/experimental/DIC/images"
    sem_image_dir = "/workspace/data/experimental/PostMortem/images"
    ensure_directories(dic_image_dir, sem_image_dir)
    generate_speckle_images(dic_image_dir)
    generate_sem_placeholder_images(sem_image_dir)
    print("Generated speckle and SEM placeholder images.")


if __name__ == "__main__":
    main()

