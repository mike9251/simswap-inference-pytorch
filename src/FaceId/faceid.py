import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from typing import Iterable, Union
from pathlib import Path


class FaceId(torch.nn.Module):
    def __init__(
        self, model_path: Path, device: str, input_shape: Iterable[int] = (112, 112)
    ):
        super().__init__()

        self.input_shape = input_shape
        self.net = torch.load(model_path, map_location=torch.device("cpu"))
        self.net.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        for n, p in self.net.named_parameters():
            assert (
                not p.requires_grad
            ), f"Parameter {n}: requires_grad: {p.requires_grad}"

        self.device = torch.device(device)
        self.to(self.device)

    def forward(
        self, img_id: Union[np.ndarray, Iterable[np.ndarray]], normalize: bool = True
    ) -> torch.Tensor:
        if isinstance(img_id, Iterable):
            img_id = [self.transform(x) for x in img_id]
            img_id = torch.stack(img_id, dim=0)
        else:
            img_id = self.transform(img_id)
            img_id = img_id.unsqueeze(0)

        img_id = img_id.to(self.device)

        img_id_112 = F.interpolate(img_id, size=self.input_shape)
        latent_id = self.net(img_id_112)
        return F.normalize(latent_id, p=2, dim=1) if normalize else latent_id
