import cv2
import numpy as np
import os
from PIL import Image
from src.data.degradation import Degradation


def main():
    sample_image_path = "src/data/sample/england7_jpg.rf.3fb28086e1b1b76ce235216eaf86a78a.jpg"
    store_path = "src/data/new_domain_samples"
    os.makedirs(store_path,exist_ok=True)
    image = cv2.imread(sample_image_path)
    scales = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    sigmas = [0,20,30,40,50,60]

    for scale,sigma in zip(scales,sigmas):
        degraded = Degradation.nearest_neighbor_gaussian_noise(image,scale=scale,sigma=sigma)
        path = os.path.join(store_path, f"Scale:{scale} Sigma{sigma}.png")
        cv2.imwrite(path,degraded)
    
    # Make Row of 6 images into 1 .png
    images=[]
    for img in reversed(sorted(os.listdir(store_path))):
        img_path = os.path.join(store_path,img)
        images.append(Image.open(img_path))
    height = max(image.height for image in images)
    width = sum(image.width for image in images)
    new_image = Image.new('RGB', (width, height))


    offset = 0
    for img in images:
        new_image.paste(img, (offset, 0))
        offset += img.width
    path = os.path.join(store_path, "grid.png")
    new_image.save(path)




if __name__ == "__main__":
    main()