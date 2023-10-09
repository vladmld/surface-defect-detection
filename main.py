"""Surface detection"""
import os

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

BASE_PATH_TO_DATA = "data"


def get_all_file_paths(directory: str, file_extension: str | None = None) -> list[str]:
    """Get all file paths from directory"""
    file_paths: list[str] = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file_extension is None or file_extension.endswith(file_extension):
                file_paths.append(os.path.join(root, file))

    return file_paths


def pre_process_image(file_path: str) -> NDArray[np.float32]:
    """Test"""
    img = cv2.imread(file_path)

    img_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img_normalized


def process_images(directory: str) -> NDArray[np.float32]:
    """Process images from directory"""
    images_paths = get_all_file_paths(directory)
    processed_images = np.zeros((len(images_paths), 227, 227, 3), dtype=np.float32)

    for i in tqdm(range(len(images_paths)), desc=f"Processing images from {directory}"):
        image_path = images_paths[i]
        processed_images[i] = pre_process_image(image_path)

    return processed_images


def pre_process():
    """Preprocesses the list of images"""
    print("Starting preprocessing ...")

    defect_images = process_images(os.path.join(BASE_PATH_TO_DATA, "defect"))
    no_defect_images = process_images(os.path.join(BASE_PATH_TO_DATA, "no_defect"))


def main():
    """The main"""
    # data loading & pre process
    pre_process()


if __name__ == "__main__":
    main()
