from typing import NamedTuple, Optional, Tuple

from insightface.model_zoo import model_zoo
import numpy as np
from pathlib import Path


class Detection(NamedTuple):
    bbox: Optional[np.ndarray]
    score: Optional[np.ndarray]
    key_points: Optional[np.ndarray]


class FaceDetector:
    def __init__(self, model_path: Path, ctx_id: int, det_thresh: float = 0.5, det_size: Tuple[int, int] = (640, 640),
                 mode: str = 'None'):
        self.det_thresh = det_thresh
        self.mode = mode
        self.handler = model_zoo.get_model(model_path)
        self.handler.prepare(ctx_id, input_size=det_size)

    def __call__(self, img: np.ndarray, max_num: int = 0) -> Detection:
        bboxes, kpss = self.handler.detect(img,
                                           threshold=self.det_thresh,
                                           max_num=max_num,
                                           metric='default')
        if bboxes.shape[0] == 0:
            return Detection(None, None, None)

        return Detection(bboxes[..., :-1], bboxes[..., -1], kpss)


if __name__ == "__main__":
    import cv2
    from SimSwapInference.FaceAlign.face_align import align_face

    det = FaceDetector(
        "C:\\Users\\petrush\\Downloads\\SimSwap\\insightface_func\\models\\antelope\\scrfd_10g_bnkps.onnx",
        ctx_id=0, det_thresh=0.5, det_size=(640, 640), mode="ffhq")

    img_path = "C:\\Users\\petrush\\Downloads\\SimSwap\\demo_file\\multi_people.jpg"
    crop_size = 224
    img = cv2.imread(img_path)

    detection = det(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    best_score = np.argmax(detection.score)

    align_imgs, transforms = align_face(img, detection.key_points, crop_size)

    cv2.imwrite("img_a.jpg", align_imgs[0])
    cv2.imwrite("img_a1.jpg", align_imgs[1])
    cv2.imwrite("img_a2.jpg", align_imgs[2])

    print("")
