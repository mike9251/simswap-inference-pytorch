import torch
import torch.nn as nn


class BlendModule(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()

        self.model = torch.jit.load(model_path).to(device)

    def forward(self, swap, mask, att_img):
        return self.model(swap, mask, att_img)
