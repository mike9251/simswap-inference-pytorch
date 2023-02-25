import numpy as np
import torch
import torch.nn.functional as F
from typing import Iterable, Tuple, Union
from pathlib import Path
from torchvision import transforms
import kornia
from omegaconf import DictConfig

from src.FaceDetector.face_detector import Detection
from src.FaceAlign.face_align import align_face, inverse_transform_batch
from src.PostProcess.utils import SoftErosion
from src.model_loader import get_model
from src.Misc.types import CheckpointType, FaceAlignmentType
from src.Misc.utils import tensor2img


class SimSwap:
    def __init__(
        self,
        config: DictConfig,
        id_image: Union[np.ndarray, None] = None,
        specific_image: Union[np.ndarray, None] = None,
    ):

        self.id_image: Union[np.ndarray, None] = id_image
        self.id_latent: Union[torch.Tensor,  None] = None
        self.specific_id_image: Union[np.ndarray,  None] = specific_image
        self.specific_latent: Union[torch.Tensor,  None] = None

        self.use_mask: Union[bool, None] = True
        self.crop_size: Union[int, None] = None
        self.checkpoint_type: Union[CheckpointType,  None] = None
        self.face_alignment_type: Union[FaceAlignmentType,  None] = None
        self.smooth_mask_iter: Union[int,  None] = None
        self.smooth_mask_kernel_size: Union[int,  None] = None
        self.smooth_mask_threshold: Union[float,  None] = None
        self.face_detector_threshold: Union[float,  None] = None
        self.specific_latent_match_threshold: Union[float,  None] = None
        self.device = torch.device(config.device)

        self.set_parameters(config)

        # For BiSeNet and for official_224 SimSwap
        self.to_tensor_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # For SimSwap models trained with the updated code
        self.to_tensor = transforms.ToTensor()

        self.face_detector = get_model(
            "face_detector",
            device=self.device,
            load_state_dice=False,
            model_path=Path(config.face_detector_weights),
            det_thresh=self.face_detector_threshold,
            det_size=(640, 640),
            mode="ffhq",
        )

        self.face_id_net = get_model(
            "arcface",
            device=self.device,
            load_state_dice=False,
            model_path=Path(config.face_id_weights),
        )

        self.bise_net = get_model(
            "parsing_model",
            device=self.device,
            load_state_dice=True,
            model_path=Path(config.parsing_model_weights),
            n_classes=19,
        )

        gen_model = "generator_512" if self.crop_size == 512 else "generator_224"
        self.simswap_net = get_model(
            gen_model,
            device=self.device,
            load_state_dice=True,
            model_path=Path(config.simswap_weights),
            input_nc=3,
            output_nc=3,
            latent_size=512,
            n_blocks=9,
            deep=True if self.crop_size == 512 else False,
            use_last_act=True
            if self.checkpoint_type == CheckpointType.OFFICIAL_224
            else False,
        )

        self.blend = get_model(
            "blend_module",
            device=self.device,
            load_state_dice=False,
            model_path=Path(config.blend_module_weights)
        )

        self.enhance_output = config.enhance_output
        if config.enhance_output:
            self.gfpgan_net = get_model(
                "gfpgan",
                device=self.device,
                load_state_dice=True,
                model_path=Path(config.gfpgan_weights)
            )

    def set_parameters(self, config) -> None:
        self.set_crop_size(config.crop_size)
        self.set_checkpoint_type(config.checkpoint_type)
        self.set_face_alignment_type(config.face_alignment_type)
        self.set_face_detector_threshold(config.face_detector_threshold)
        self.set_specific_latent_match_threshold(config.specific_latent_match_threshold)
        self.set_smooth_mask_kernel_size(config.smooth_mask_kernel_size)
        self.set_smooth_mask_threshold(config.smooth_mask_threshold)
        self.set_smooth_mask_iter(config.smooth_mask_iter)

    def set_crop_size(self, crop_size: int) -> None:
        if crop_size < 0:
            raise "Invalid crop_size! Must be a positive value."

        self.crop_size = crop_size

    def set_checkpoint_type(self, checkpoint_type: str) -> None:
        type = CheckpointType(checkpoint_type)
        if type not in (CheckpointType.OFFICIAL_224, CheckpointType.UNOFFICIAL):
            raise "Invalid checkpoint_type! Must be one of the predefined values."

        self.checkpoint_type = type

    def set_face_alignment_type(self, face_alignment_type: str) -> None:
        type = FaceAlignmentType(face_alignment_type)
        if type not in (
            FaceAlignmentType.FFHQ,
            FaceAlignmentType.DEFAULT,
        ):
            raise "Invalid face_alignment_type! Must be one of the predefined values."

        self.face_alignment_type = type

    def set_face_detector_threshold(self, face_detector_threshold: float) -> None:
        if face_detector_threshold < 0.0 or face_detector_threshold > 1.0:
            raise "Invalid face_detector_threshold! Must be a positive value in range [0.0...1.0]."

        self.face_detector_threshold = face_detector_threshold

    def set_specific_latent_match_threshold(
        self, specific_latent_match_threshold: float
    ) -> None:
        if specific_latent_match_threshold < 0.0:
            raise "Invalid specific_latent_match_th! Must be a positive value."

        self.specific_latent_match_threshold = specific_latent_match_threshold

    def re_initialize_soft_mask(self):
        self.smooth_mask = SoftErosion(kernel_size=self.smooth_mask_kernel_size,
                                       threshold=self.smooth_mask_threshold,
                                       iterations=self.smooth_mask_iter).to(self.device)

    def set_smooth_mask_kernel_size(self, smooth_mask_kernel_size: int) -> None:
        if smooth_mask_kernel_size < 0:
            raise "Invalid smooth_mask_kernel_size! Must be a positive value."
        smooth_mask_kernel_size += 1 if smooth_mask_kernel_size % 2 == 0 else 0
        self.smooth_mask_kernel_size = smooth_mask_kernel_size
        self.re_initialize_soft_mask()

    def set_smooth_mask_threshold(self, smooth_mask_threshold: int) -> None:
        if smooth_mask_threshold < 0 or smooth_mask_threshold > 1.0:
            raise "Invalid smooth_mask_threshold! Must be within 0...1 range."
        self.smooth_mask_threshold = smooth_mask_threshold
        self.re_initialize_soft_mask()

    def set_smooth_mask_iter(self, smooth_mask_iter: float) -> None:
        if smooth_mask_iter < 0:
            raise "Invalid smooth_mask_iter! Must be a positive value.."
        self.smooth_mask_iter = smooth_mask_iter
        self.re_initialize_soft_mask()

    def run_detect_align(self, image: np.ndarray, for_id: bool = False) -> Tuple[Union[Iterable[np.ndarray], None],
                                                                                 Union[Iterable[np.ndarray], None],
                                                                                 np.ndarray]:
        detection: Detection = self.face_detector(image)

        if detection.bbox is None:
            if for_id:
                raise "Can't detect a face! Please change the ID image!"
            return None, None, detection.score

        kps = detection.key_points

        if for_id:
            max_score_ind = np.argmax(detection.score, axis=0)
            kps = detection.key_points[max_score_ind]
            kps = kps[None, ...]

        align_imgs, transforms = align_face(
            image,
            kps,
            crop_size=self.crop_size,
            mode="ffhq"
            if self.face_alignment_type == FaceAlignmentType.FFHQ
            else "none",
        )

        return align_imgs, transforms, detection.score

    def __call__(self, att_image: np.ndarray) -> np.ndarray:
        if self.id_latent is None:
            align_id_imgs, id_transforms, _ = self.run_detect_align(
                self.id_image, for_id=True
            )
            # normalize=True, because official SimSwap model trained with normalized id_lattent
            self.id_latent: torch.Tensor = self.face_id_net(
                align_id_imgs, normalize=True
            )

        if self.specific_id_image is not None and self.specific_latent is None:
            align_specific_imgs, specific_transforms, _ = self.run_detect_align(
                self.specific_id_image, for_id=True
            )
            self.specific_latent: torch.Tensor = self.face_id_net(
                align_specific_imgs, normalize=False
            )

        # for_id=False, because we want to get all faces
        align_att_imgs, att_transforms, att_detection_score = self.run_detect_align(
            att_image, for_id=False
        )

        if align_att_imgs is None and att_transforms is None:
            return att_image

        # Select specific crop from the target image
        if self.specific_latent is not None:
            att_latent: torch.Tensor = self.face_id_net(align_att_imgs, normalize=False)
            latent_dist = torch.mean(
                F.mse_loss(
                    att_latent,
                    self.specific_latent.repeat(att_latent.shape[0], 1),
                    reduction="none",
                ),
                dim=-1,
            )

            att_detection_score = torch.tensor(
                att_detection_score, device=latent_dist.device
            )

            min_index = torch.argmin(latent_dist * att_detection_score)
            min_value = latent_dist[min_index]

            if min_value < self.specific_latent_match_threshold:
                align_att_imgs = [align_att_imgs[min_index]]
                att_transforms = [att_transforms[min_index]]
            else:
                return att_image

        swapped_img: torch.Tensor = self.simswap_net(align_att_imgs, self.id_latent)

        if self.enhance_output:
            swapped_img = self.gfpgan_net.enhance(swapped_img, weight=0.5)

        # Put all crops/transformations into a batch
        align_att_img_batch_for_parsing_model: torch.Tensor = torch.stack(
            [self.to_tensor_normalize(x) for x in align_att_imgs], dim=0
        )
        align_att_img_batch_for_parsing_model = (
            align_att_img_batch_for_parsing_model.to(self.device)
        )

        att_transforms: torch.Tensor = torch.stack(
            [torch.tensor(x).float() for x in att_transforms], dim=0
        )
        att_transforms = att_transforms.to(self.device, non_blocking=True)

        align_att_img_batch: torch.Tensor = torch.stack(
            [self.to_tensor(x) for x in align_att_imgs], dim=0
        )
        align_att_img_batch = align_att_img_batch.to(self.device, non_blocking=True)

        # Get face masks for the attribute image
        face_mask, ignore_mask_ids = self.bise_net.get_mask(
            align_att_img_batch_for_parsing_model, self.crop_size
        )

        inv_att_transforms: torch.Tensor = inverse_transform_batch(att_transforms)

        soft_face_mask, _ = self.smooth_mask(face_mask)

        swapped_img[ignore_mask_ids, ...] = align_att_img_batch[ignore_mask_ids, ...]

        frame_size = (att_image.shape[0], att_image.shape[1])

        att_image = self.to_tensor(att_image).to(self.device, non_blocking=True).unsqueeze(0)

        target_image = kornia.geometry.transform.warp_affine(
            swapped_img,
            inv_att_transforms,
            frame_size,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
            fill_value=torch.zeros(3),
        )

        soft_face_mask = kornia.geometry.transform.warp_affine(
            soft_face_mask,
            inv_att_transforms,
            frame_size,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
            fill_value=torch.zeros(3),
        )

        result = self.blend(target_image, soft_face_mask, att_image)

        return tensor2img(result)
