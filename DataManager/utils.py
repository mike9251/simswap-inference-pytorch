import cv2
import numpy as np
from pathlib import Path
from typing import Union


def imread_rgb(img_path: Union[str, Path]) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)


def imwrite_rgb(img_path: Union[str, Path], img):
    return cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
