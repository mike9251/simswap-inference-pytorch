from typing import NamedTuple, Optional, Tuple

from insightface.model_zoo import model_zoo
import numpy as np
from pathlib import Path


class Detection(NamedTuple):
    bbox: Optional[np.ndarray]
    score: Optional[np.ndarray]
    key_points: Optional[np.ndarray]


class FaceDetector:
    def __init__(
        self,
        model_path: Path,
        det_thresh: float = 0.5,
        det_size: Tuple[int, int] = (640, 640),
        mode: str = "None",
        device: str = "cpu",
    ):
        self.det_thresh = det_thresh
        self.mode = mode
        self.device = device
        self.handler = model_zoo.get_model(str(model_path))
        ctx_id = -1 if device == "cpu" else 0
        self.handler.prepare(ctx_id, input_size=det_size)

    def __call__(self, img: np.ndarray, max_num: int = 0) -> Detection:
        bboxes, kpss = self.handler.detect(
            img, threshold=self.det_thresh, max_num=max_num, metric="default"
        )
        if bboxes.shape[0] == 0:
            return Detection(None, None, None)

        return Detection(bboxes[..., :-1], bboxes[..., -1], kpss)
