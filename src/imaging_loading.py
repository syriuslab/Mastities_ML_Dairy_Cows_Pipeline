"""Helpers for loading thermal images for the mastitis project."""

from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def load_images_from_dir(images_dir: str, img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    imgs: List[np.ndarray] = []
    for img_path in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    if not imgs:
        raise RuntimeError(f"No images found in {images_dir}. Please add your thermal images.")

    return np.stack(imgs, axis=0)
