from src.DataManager.base import BaseDataManager
from src.DataManager.utils import imwrite_rgb

import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import Optional, Union

from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class VideoDataManager(BaseDataManager):
    def __init__(self, src_data: Path, output_dir: Path, clean_work_dir: bool = False):
        self.video_handle: Optional[cv2.VideoCapture] = None
        self.audio_handle: Optional[AudioFileClip] = None

        self.output_dir = output_dir
        self.output_img_dir = output_dir / "img"
        self.output_dir.mkdir(exist_ok=True)
        self.output_img_dir.mkdir(exist_ok=True)
        self.video_name = None
        self.clean_work_dir = clean_work_dir

        if src_data.is_file():
            self.video_name = "swap_" + src_data.name

            if VideoFileClip(str(src_data)).audio is not None:
                self.audio_handle = AudioFileClip(str(src_data))

            self.video_handle = cv2.VideoCapture(str(src_data))
            self.video_handle.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.frame_count = int(self.video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_handle.get(cv2.CAP_PROP_FPS)

        self.last_idx = -1

        assert self.video_handle, "Video file must be specified!"

    def __len__(self):
        return self.frame_count

    def get(self) -> np.ndarray:
        img: Union[None, np.ndarray] = None

        while img is None and self.last_idx < self.frame_count:
            status, img = self.video_handle.read()
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
        image_filenames = [str(x) for x in sorted(self.output_img_dir.glob("*.jpg"))]
        clip = ImageSequenceClip(image_filenames, fps=self.fps)

        if self.audio_handle is not None:
            clip = clip.set_audio(self.audio_handle)

        clip.write_videofile(str(self.output_dir / self.video_name))

        if self.clean_work_dir:
            shutil.rmtree(self.output_img_dir, ignore_errors=True)
