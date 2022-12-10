from src.DataManager.base import BaseDataManager
from src.DataManager.utils import imwrite_rgb

import numpy as np
from pathlib import Path
from typing import Optional

import cv2
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class VideoDataManager(BaseDataManager):
    def __init__(self, src_data: Path, output_dir: Path):
        self.video_handle: Optional[VideoFileClip] = None
        self.audio_handle: Optional[AudioFileClip] = None

        self.output_dir = output_dir
        self.output_img_dir = output_dir / "img"
        self.output_dir.mkdir(exist_ok=True)
        self.output_img_dir.mkdir(exist_ok=True)
        self.video_name = None

        if src_data.is_file():
            self.video_name = "swap_" + src_data.name

            self.audio_handle = AudioFileClip(str(src_data))
            self.video_handle = VideoFileClip(str(src_data))
            self.fps = self.video_handle.reader.fps
            self.frame_count = self.video_handle.reader.nframes
            self.data_iterator = zip(range(self.frame_count), self.video_handle.iter_frames())

        self.last_idx = -1

        assert self.video_handle, "Video file must be specified!"

    def __len__(self):
        return self.frame_count

    def get(self) -> np.ndarray:
        self.last_idx, img = next(self.data_iterator)
        return img

    def save(self, img: np.ndarray):
        filename = "frame_{:0>7d}.jpg".format(self.last_idx)
        imwrite_rgb(self.output_img_dir / filename, img)

        if (self.frame_count - 1) == self.last_idx:
            self._close()

    def _close(self):
        image_filenames = [str(x) for x in sorted(self.output_img_dir.glob("*.jpg"))]
        clip = ImageSequenceClip(image_filenames, fps=self.fps)

        clip = clip.set_audio(self.audio_handle)

        clip.write_videofile(str(self.output_dir / self.video_name), audio_codec="aac")
