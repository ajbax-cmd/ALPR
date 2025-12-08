import cv2
import numpy as np
import os
from PIL import Image


def main():
    dir = os.path.join(os.getcwd(), "inference_samples")
    if os.path.exists(dir):
        images=[]
        for img in reversed(sorted(os.listdir(dir))):
            img_path = os.path.join(dir,img)
            images.append(Image.open(img_path))
    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    new_image = Image.new('RGB', (width, height))
    offset = 0
    for img in images:
        new_image.paste(img, (offset, 0))
        offset += img.width
    path = os.path.join(dir, "inference_grid.png")
    new_image.save(path)

if __name__ == "__main__":
    main()