import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Iterable, Tuple
from pathlib import Path
from torchvision import transforms
import kornia
import time
from omegaconf import DictConfig

from src.FaceDetector.face_detector import FaceDetector, Detection
from src.FaceAlign.face_align import align_face, inverse_transform_batch
from src.FaceId.faceid import FaceId
from src.PostProcess.ParsingModel.model import BiSeNet
from src.PostProcess.utils import SoftErosion
from src.Generator.fs_networks_fix import Generator_Adain_Upsample as Generator_Adain_Upsample_224
from src.Generator.fs_networks_512 import Generator_Adain_Upsample as Generator_Adain_Upsample_512


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


class SimSwap:
    def __init__(self,
                 config: DictConfig,
                 id_image: np.ndarray,
                 specific_image: Optional[np.ndarray] = None,
                 use_mask: bool = True,
                 crop_size: int = 224,
                 device: str = 'cpu'):

        self.id_image: np.ndarray = id_image
        self.id_latent = None
        self.specific_id_image: Optional[np.ndarray] = specific_image
        self.specific_latent = None
        self.specific_latent_match_th = 0.03

        self.use_mask = use_mask
        self.crop_size = crop_size
        self.checkpoint_type = config.checkpoint_type
        self.face_alignment_type = config.face_alignment_type
        self.device = torch.device(device)

        # For BiSeNet and for official_224 SimSwap
        self.to_tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # For SimSwap models trained with the updated code
        self.to_tensor = transforms.ToTensor()

        self.face_detector = FaceDetector(
            Path(config.face_detector_weights),
            det_thresh=0.5, det_size=(640, 640), mode="ffhq", device=device)

        self.face_id_net = FaceId(Path(config.face_id_weights)).to(self.device)

        self.bise_net = BiSeNet(n_classes=19)
        bise_net_clpt = torch.load(Path(config.parsing_model_weights))
        self.bise_net.load_state_dict(bise_net_clpt)
        self.bise_net = self.bise_net.to(self.device)
        self.bise_net.eval()

        self.simswap_net = Generator_Adain_Upsample_224(input_nc=3, output_nc=3, latent_size=512, n_blocks=9,
                                                        deep=True if self.crop_size == 512 else False,
                                                        checkpoint_type=self.checkpoint_type)

        # if crop_size == 224:
        #     self.simswap_net = Generator_Adain_Upsample_224(input_nc=3, output_nc=3, latent_size=512, n_blocks=9,
        #                                                     deep=False)
        # else:
        #     # Bottleneck is different. In the checkpoint there are BN layers, but their parameters seem to be not optimized
        #     # running_mean = 0, running_var = 1, weights = 1, bias = 0
        #     # self.simswap_net = Generator_Adain_Upsample_512(input_nc=3, output_nc=3, latent_size=512, n_blocks=9,
        #     #                                                 deep=False)
        #     self.simswap_net = Generator_Adain_Upsample_512(input_nc=3, output_nc=3, latent_size=512, n_blocks=9,
        #                                                     deep=False)

        simswap_net_ckpt = torch.load(Path(config.simswap_weights))
        self.simswap_net.load_state_dict(simswap_net_ckpt)
        self.simswap_net = self.simswap_net.to(self.device)
        self.simswap_net.eval()

        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).to(self.device)

    def run_detect_align(self, image: np.ndarray, for_id: bool = False) -> Tuple[
        Iterable[np.ndarray], Iterable[np.ndarray]]:
        detection: Detection = self.face_detector(image)

        kps = detection.key_points

        if for_id:
            max_score_ind = np.argmax(detection.score, axis=0)
            kps = detection.key_points[max_score_ind]
            kps = kps[None, ...]

        align_imgs, transforms = align_face(image, kps, crop_size=self.crop_size,
                                            mode=self.face_alignment_type)

        return align_imgs, transforms

    def __call__(self, att_image: np.ndarray) -> np.ndarray:
        if self.id_latent is None:
            align_id_imgs, id_transforms = self.run_detect_align(self.id_image, for_id=True)
            # normalize=True, because official SimSwap model trained with normalized id_lattent
            self.id_latent: torch.Tensor = self.face_id_net(align_id_imgs, normalize=True)

        if self.specific_id_image is not None and self.specific_latent is None:
            align_specific_imgs, specific_transforms = self.run_detect_align(self.specific_id_image, for_id=True)
            self.specific_latent: torch.Tensor = self.face_id_net(align_specific_imgs, normalize=False)

        # for_id=False, because we want to get all faces
        align_att_imgs, att_transforms = self.run_detect_align(att_image, for_id=False)

        # Select specific crop from the target image
        if self.specific_latent is not None:
            att_latent: torch.Tensor = self.face_id_net(align_att_imgs, normalize=False)
            latent_dist = torch.mean(
                F.mse_loss(att_latent, self.specific_latent.repeat(att_latent.shape[0], 1), reduction='none'), dim=-1)

            min_index = torch.argmin(latent_dist)
            min_value = latent_dist[min_index]

            if min_value < self.specific_latent_match_th:
                align_att_imgs = [align_att_imgs[min_index]]
                att_transforms = [att_transforms[min_index]]

        swapped_img: torch.Tensor = self.simswap_net(align_att_imgs, self.id_latent)

        # Put all crops/transformations into a batch
        align_att_img_batch_for_parsing_model: torch.Tensor = torch.stack(
            [self.to_tensor_normalize(x) for x in align_att_imgs], dim=0)
        align_att_img_batch_for_parsing_model = align_att_img_batch_for_parsing_model.to(self.device)

        att_transforms: torch.Tensor = torch.stack([torch.tensor(x) for x in att_transforms], dim=0)
        att_transforms = att_transforms.to(self.device)

        align_att_img_batch: torch.Tensor = torch.stack([self.to_tensor(x) for x in align_att_imgs], dim=0)
        align_att_img_batch = align_att_img_batch.to(self.device)

        img_white = torch.zeros_like(align_att_img_batch) + 255

        inv_att_transforms: torch.Tensor = inverse_transform_batch(att_transforms)

        # Get face masks for the attribute image
        face_mask = self.bise_net.get_mask(align_att_img_batch_for_parsing_model, self.crop_size)

        soft_face_mask, _ = self.smooth_mask(face_mask)

        # Only take face area from the swapped image
        swapped_img = swapped_img * soft_face_mask + align_att_img_batch * (1 - soft_face_mask)

        frame_size = (att_image.shape[0], att_image.shape[1])

        # Place swapped faces and masks where they should be in the original frame
        target_image = kornia.geometry.transform.warp_affine(swapped_img.double(), inv_att_transforms,
                                                             frame_size,
                                                             mode='bilinear', padding_mode='zeros',
                                                             align_corners=True, fill_value=torch.zeros(3))

        img_mask = kornia.geometry.transform.warp_affine(img_white.double(), inv_att_transforms, frame_size,
                                                         mode='bilinear', padding_mode='zeros',
                                                         align_corners=True, fill_value=torch.zeros(3))

        img_mask[img_mask > 20] = 255

        # numpy postprocessing
        # Collect masks for all crops
        img_mask = torch.sum(img_mask, dim=0, keepdim=True)

        # Get np.ndarray with range [0...255]
        img_mask = tensor2img(img_mask / 255.0)

        # Make it configurable
        erosion_kernel_size = 40
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)

        kernel_size = (41, 41)
        img_mask = cv2.GaussianBlur(img_mask, kernel_size, 0)

        # Collect all swapped crops
        target_image = torch.sum(target_image, dim=0, keepdim=True)
        target_image = tensor2img(target_image)

        img_mask = img_mask // 255

        result = img_mask * target_image + (1 - img_mask) * att_image

        # # torch postprocessing
        # # faster but Erosion with 40x40 kernel requires too much memory and causes OOM.
        # # Using smaller kernel sometimes causes visual artifacts along the mask border
        #
        # # Collect masks for all crops
        # img_mask = torch.sum(img_mask, dim=0, keepdim=True)
        #
        # img_mask /= 255
        # # cv2.imwrite("img_mask.jpg", tensor2img(img_mask))
        #
        # kernel = torch.ones((4, 4), dtype=torch.int, device=img_mask.device)
        # img_mask = kornia.morphology.erosion(img_mask, kernel, structuring_element=None, origin=None, border_type='geodesic', border_value=0.0, max_val=1.0, engine='unfold')
        # # cv2.imwrite("img_mask_erode.jpg", tensor2img(img_mask))
        #
        # kernel_size = (41, 41)
        # sigma = 0.05
        # # Should be https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        # # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        # # sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
        # img_mask = kornia.filters.gaussian_blur2d(img_mask, kernel_size, (sigma, sigma), border_type='reflect', separable=True)
        # # cv2.imwrite("img_mask_gaus.jpg", tensor2img(img_mask))
        #
        # # Collect all swapped crops
        # target_image = torch.sum(target_image, dim=0, keepdim=True)
        #
        # result = tensor2img(img_mask * target_image + (1 - img_mask) * self.to_tensor(att_image).to(self.device))

        return result
