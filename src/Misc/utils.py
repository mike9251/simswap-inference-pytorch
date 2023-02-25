import torch
import numpy as np
import cv2


def tensor2img_denorm(tensor):
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    tensor = std * tensor.detach().cpu() + mean
    img = tensor.numpy()
    img = img.transpose(0, 2, 3, 1)[0]
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    return img


def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy()
    img = tensor.transpose(0, 2, 3, 1)[0]
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    return img


def show_tensor(tensor, name):
    img = cv2.cvtColor(tensor2img(tensor), cv2.COLOR_RGB2BGR)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey()
