from pathlib import Path
import time
from typing import Optional
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
import numpy as np

from SimSwapInference.src.simswap import SimSwap
from SimSwapInference.src.DataManager.ImageDataManager import ImageDataManager
from SimSwapInference.src.DataManager.VideoDataManager import VideoDataManager
from SimSwapInference.src.DataManager.utils import imread_rgb


class Application:
    def __init__(self, config: DictConfig):

        id_image_path = Path(config.data.id_image)
        specific_id_image_path = Path(config.data.specific_id_image)
        att_image_path = Path(config.data.att_image)
        att_video_path = Path(config.data.att_video)
        output_dir = Path(config.data.output_dir)

        device = config.pipeline.device

        crop_size = config.pipeline.crop_size
        use_mask = True

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
                             specific_image=self.specific_id_image,
                             use_mask=use_mask,
                             crop_size=crop_size,
                             device=device)

    def run(self):
        for i in tqdm(range(len(self.data_manager))):

            att_img = self.data_manager.get()

            output = self.model(att_img)

            self.data_manager.save(output)

        self.data_manager.close()


@hydra.main(config_path="configs/", config_name="run_image.yaml")
def main(config: DictConfig):

    app = Application(config)

    app.run()


if __name__ == "__main__":
    main()
