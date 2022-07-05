import cv2
import numpy as np
import torch
from typing import Optional, Iterable, Tuple
from pathlib import Path

from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from SimSwapInference.FaceDetector.face_detector import FaceDetector, Detection
from SimSwapInference.FaceAlign.face_align import align_face, inverse_transform
from SimSwapInference.FaceId.faceid import FaceId
from SimSwapInference.PostProcess.ParsingModel.model import BiSeNet
from SimSwapInference.PostProcess.utils import postprocess, postprocess, SoftErosion


def imread_rgb(img_path):
    return cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)


class Application:
    def __init__(self,
                 id_image_path: Path,
                 att_image_path: Optional[Path] = None,
                 att_video_path: Optional[Path] = None,
                 specific_id_image_path: Optional[Path] = None,
                 output_dir: Path = '..',
                 use_mask: bool = True,
                 crop_size: int = 224,
                 device: torch.device = torch.device('cpu')):
        assert id_image_path, f"ID image path must be specified!"
        assert id_image_path.exists(), f"Can't find {id_image_path} file!"

        assert att_image_path or att_video_path, f"Attribute image or video path must be specified!"
        assert att_image_path.exists() or att_video_path.exists(), f"Can't find attribute file!"

        self.id_image: Optional[np.ndarray] = imread_rgb(id_image_path)
        self.id_latent: Optional[torch.Tensor] = None
        self.att_image: Optional[np.ndarray] = imread_rgb(att_image_path) if att_image_path else None
        self.specific_id_image: Optional[np.ndarray] = imread_rgb(
            specific_id_image_path) if specific_id_image_path else None
        self.att_video_handler: Optional[cv2.VideoCapture] = None
        self.att_video_handler: Optional[AudioFileClip] = None

        if att_video_path:
            with_audio = True if VideoFileClip(att_video_path).audio else False
            self.att_audio_clip = AudioFileClip(att_video_path) if with_audio else None
            self.att_video_handler = cv2.VideoCapture(att_video_path)

            self.frame_count = int(self.att_video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
            # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.att_video_handler.get(cv2.CAP_PROP_FPS)

        self.use_mask = use_mask
        self.crop_size = crop_size
        self.device = device

        self.face_detector = FaceDetector(
            "C:\\Users\\petrush\\Downloads\\SimSwap\\insightface_func\\models\\antelope\\scrfd_10g_bnkps.onnx",
            ctx_id=0, det_thresh=0.5, det_size=(640, 640), mode="ffhq")

        self.face_id_net = FaceId("C:\\Users\\petrush\\Downloads\\SimSwap\\arcface_model\\arcface_net.jit").to(device)
        self.bise_net = BiSeNet(n_classes=19).to(device)
        self.simswap_net = None

        self.smooth_mask = SoftErosion().to(device)

    def run_detect_align(self, image):
        detection: Detection = self.face_detector(image)
        align_att_imgs, transforms = align_face(image, detection.key_points, crop_size=self.crop_size)
        return align_att_imgs, transforms

    def run_detect_align_id(self, image) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray], torch.Tensor]:
        align_att_imgs, transforms = self.run_detect_align(image)
        # Make a batch from the list align_att_imgs
        id_latent = self.face_id_net(align_att_imgs)
        return align_att_imgs, transforms, id_latent

    def run_frame(self, att_img):
        pass

    def run_single_image(self):
        align_id_imgs, id_transforms, id_latent = self.run_detect_align_id(self.id_image)

        align_att_imgs, att_transforms = self.run_detect_align(self.att_image)

        swapped_img = self.simswap_net(align_att_imgs, id_latent)

        # if
        # for frame_index in range(self.frame_count):
        #     ret, frame = self.att_video_handler.read()
        #     if ret:
        #         detect_results = detect_model.get(frame, crop_size)



if __name__ == "__main__":
    app = Application(id_image_path=Path("C:\\Users\\petrush\\Downloads\\SimSwap\\demo_file\\Iron_man.jpg"),
                      att_image_path=Path("C:\\Users\\petrush\\Downloads\\SimSwap\\demo_file\\specific1.png"),
                      output_dir=Path(".."))
