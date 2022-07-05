from SimSwapInference.DataManager.base import BaseDataManager
from SimSwapInference.DataManager.utils import imwrite_rgb

import numpy as np
from pathlib import Path
from typing import Optional, Union

import cv2
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


class VideoDataManager(BaseDataManager):
    def __init__(self, data_path: Path, output_dir: Path):
        self.video_handler: Optional[cv2.VideoCapture] = None
        self.audio_handler: Optional[AudioFileClip] = None

        self.output_dir = output_dir
        self.output_img_dir = output_dir / 'img'
        self.output_dir.mkdir(exist_ok=True)
        self.output_img_dir.mkdir(exist_ok=True)
        self.video_name = None

        if data_path.is_file():
            self.video_name = 'swap_' + data_path.name

            self.audio_handler = AudioFileClip(str(data_path))
            self.video_handler = cv2.VideoCapture(str(data_path))

            self.frame_count = int(self.video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
            # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.video_handler.get(cv2.CAP_PROP_FPS)

        self.last_idx = -1

        assert self.video_handler, f'Video file must be specified!'

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
        filename = 'frame_{:0>7d}.jpg'.format(self.last_idx)
        imwrite_rgb(self.output_img_dir / filename, img)

    def write_video(self):
        image_filenames = [str(x) for x in sorted(self.output_img_dir.glob("*.jpg"))]
        clip = ImageSequenceClip(image_filenames, fps=self.fps)

        clip = clip.set_audio(self.audio_handler)

        clip.write_videofile(str(self.output_dir / self.video_name), audio_codec='aac')


if __name__ == "__main__":
    dm = VideoDataManager(data_path=Path(r"C:\Users\petrush\Downloads\SimSwap\demo_file\multi_people_1080p.mp4"),
                          output_dir=Path.cwd())

    for i in range(len(dm)):
        sample = dm.get()
        print(sample.shape)

        dm.save(sample)

    dm.write_video()


