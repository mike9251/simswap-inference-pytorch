import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple


class SoftErosion(torch.nn.Module):
    def __init__(
        self, kernel_size: int = 15, threshold: float = 0.6, iterations: int = 1
    ):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(
            torch.arange(0.0, kernel_size), torch.arange(0.0, kernel_size)
        )
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer("weight", kernel)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(
                x,
                F.conv2d(
                    x, weight=self.weight, groups=x.shape[1], padding=self.padding
                ),
            )
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def encode_segmentation_rgb(
    segmentation: np.ndarray, no_neck: bool = True
) -> np.ndarray:
    parse = segmentation
    # https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py
    face_part_ids = (
        [1, 2, 3, 4, 5, 6, 10, 12, 13]
        if no_neck
        else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    )
    mouth_id = 11
    # hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    # hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse == valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse == mouth_id)
    mouth_map[valid_index] = 255
    # valid_index = np.where(parse==hair_id)
    # hair_map[valid_index] = 255
    # return np.stack([face_map, mouth_map,hair_map], axis=2)
    return np.stack([face_map, mouth_map], axis=2)


def encode_segmentation_rgb_batch(
    segmentation: torch.Tensor, no_neck: bool = True
) -> torch.Tensor:
    # https://github.com/zllrunning/face-parsing.PyTorch/blob/master/prepropess_data.py
    face_part_ids = (
        [1, 2, 3, 4, 5, 6, 10, 12, 13]
        if no_neck
        else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    )
    mouth_id = 11
    # hair_id = 17
    segmentation = segmentation.int()
    face_map = torch.zeros_like(segmentation)
    mouth_map = torch.zeros_like(segmentation)
    # hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    white_tensor = face_map + 255
    for valid_id in face_part_ids:
        face_map = torch.where(segmentation == valid_id, white_tensor, face_map)
    mouth_map = torch.where(segmentation == mouth_id, white_tensor, mouth_map)

    return torch.cat([face_map, mouth_map], dim=1)


def postprocess(
    swapped_face: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    smooth_mask: torch.nn.Module,
) -> np.ndarray:
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))

    mask_tensor = (
        torch.from_numpy(target_mask.copy().transpose((2, 0, 1)))
        .float()
        .mul_(1 / 255.0)
        .cuda()
    )
    face_mask_tensor = mask_tensor[0] + mask_tensor[1]

    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis]

    result = swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:, :, ::-1]  # .astype(np.uint8)
    return result
