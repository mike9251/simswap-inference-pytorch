from src.DataManager.base import BaseDataManager
from src.DataManager.utils import imwrite_rgb

import numpy as np
from pathlib import Path
from typing import Optional, Union

import cv2
from moviepy.editor import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class VideoDataManager(BaseDataManager):
    def __init__(self, src_data: Path, output_dir: Path):
        self.video_handler: Optional[cv2.VideoCapture] = None
        self.audio_handler: Optional[AudioFileClip] = None

        self.output_dir = output_dir
        self.output_img_dir = output_dir / "img"
        self.output_dir.mkdir(exist_ok=True)
        self.output_img_dir.mkdir(exist_ok=True)
        self.video_name = None

        if src_data.is_file():
            self.video_name = "swap_" + src_data.name

            self.audio_handler = AudioFileClip(str(src_data))
            self.video_handler = cv2.VideoCapture(str(src_data))

            self.frame_count = int(self.video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
            # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.video_handler.get(cv2.CAP_PROP_FPS)

        self.last_idx = -1

        assert self.video_handler, "Video file must be specified!"

    def __len__(self):
        return self.frame_count

    def get(self) -> np.ndarray:
        img: Union[None, np.ndarray] = None

        while img is None and self.last_idx < self.frame_count:
            status, img = self.video_handler.read()
            self.last_idx += 1

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def save(self, img: np.ndarray):
        filename = "frame_{:0>7d}.jpg".format(self.last_idx)
        imwrite_rgb(self.output_img_dir / filename, img)

        if (self.frame_count - 1) == self.last_idx:
            self._close()

    def _close(self):
        self.video_handler.release()

        image_filenames = [str(x) for x in sorted(self.output_img_dir.glob("*.jpg"))]
        clip = ImageSequenceClip(image_filenames, fps=self.fps)

        clip = clip.set_audio(self.audio_handler)

        clip.write_videofile(str(self.output_dir / self.video_name), audio_codec="aac")
