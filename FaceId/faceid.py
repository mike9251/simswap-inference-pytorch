import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from typing import Iterable
from pathlib import Path


class FaceId(torch.nn.Module):
    def __init__(self, arcnet_path: Path, input_shape: Iterable[int] = (112, 112)):
        super().__init__()

        self.input_shape = input_shape
        self.net = torch.load(arcnet_path, map_location=torch.device("cpu"))
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for n, p in self.net.named_parameters():
            assert not p.requires_grad, f"Parameter {n}: requires_grad: {p.requires_grad}"

    def forward(self, img_id: np.ndarray, normalize: bool = True) -> torch.Tensor:
        img_id = self.transform(img_id)
        img_id_112 = torch.clamp(F.interpolate(img_id, size=self.input_shape, mode='bicubic'), min=0.0)
        latent_id = self.net(img_id_112)
        return F.normalize(latent_id, p=2, dim=1) if normalize else latent_id
