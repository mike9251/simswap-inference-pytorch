from pathlib import Path
import time
from typing import Optional
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
import numpy as np

from src.simswap import SimSwap
from src.DataManager.ImageDataManager import ImageDataManager
from src.DataManager.VideoDataManager import VideoDataManager
from src.DataManager.utils import imread_rgb


class Application:
    def __init__(self, config: DictConfig):

        id_image_path = Path(config.data.id_image)
        specific_id_image_path = Path(config.data.specific_id_image)
        att_image_path = Path(config.data.att_image)
        att_video_path = Path(config.data.att_video)
        output_dir = Path(config.data.output_dir)

        assert id_image_path.exists(), f"Can't find {id_image_path} file!"

        self.id_image: Optional[np.ndarray] = imread_rgb(id_image_path)
        self.specific_id_image: Optional[np.ndarray] = imread_rgb(
            specific_id_image_path) if specific_id_image_path and specific_id_image_path.is_file() else None

        self.att_image: Optional[ImageDataManager] = None
        if att_image_path and (att_image_path.is_file() or att_image_path.is_dir()):
            self.att_image: Optional[ImageDataManager] = ImageDataManager(src_data=att_image_path,
                                                                          output_dir=output_dir)

        self.att_video: Optional[VideoDataManager] = None
        if att_video_path and att_video_path.is_file():
            self.att_video: Optional[VideoDataManager] = VideoDataManager(src_data=att_video_path,
                                                                          output_dir=output_dir)

        assert not (self.att_video and self.att_image), f'Only one attribute source can be used!'

        self.data_manager = self.att_video if self.att_video else self.att_image

        self.model = SimSwap(config=config.pipeline,
                             id_image=self.id_image,
                             specific_image=self.specific_id_image)

    def run(self):
        for _ in tqdm(range(len(self.data_manager))):

            att_img = self.data_manager.get()

            output = self.model(att_img)

            self.data_manager.save(output)


@hydra.main(config_path="configs/", config_name="run_image.yaml")
def main(config: DictConfig):

    app = Application(config)

    app.run()


if __name__ == "__main__":
    main()
