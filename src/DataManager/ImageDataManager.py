from src.DataManager.base import BaseDataManager
from src.DataManager.utils import imread_rgb, imwrite_rgb

import numpy as np
from pathlib import Path


class ImageDataManager(BaseDataManager):
    def __init__(self, src_data: Path, output_dir: Path):
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.output_dir = output_dir / "img"
        self.output_dir.mkdir(exist_ok=True)

        self.data_paths = []
        if src_data.is_file():
            self.data_paths.append(src_data)
        elif src_data.is_dir():
            self.data_paths = (
                list(src_data.glob("*.jpg"))
                + list(src_data.glob("*.jpeg"))
                + list(src_data.glob("*.png"))
            )

        assert len(self.data_paths), "Data must be supplied!"

        self.data_paths_iter = iter(self.data_paths)

        self.last_idx = -1

    def __len__(self):
        return len(self.data_paths)

    def get(self) -> np.ndarray:
        img_path = next(self.data_paths_iter)
        self.last_idx += 1
        return imread_rgb(img_path)

    def save(self, img: np.ndarray):
        filename = "swap_" + Path(self.data_paths[self.last_idx]).name

        imwrite_rgb(self.output_dir / filename, img)
