from collections import namedtuple
import torch
from torch.utils import model_zoo
import requests
from tqdm import tqdm
from pathlib import Path

from src.FaceDetector.face_detector import FaceDetector
from src.FaceId.faceid import FaceId
from src.Generator.fs_networks_fix import Generator_Adain_Upsample
from src.PostProcess.ParsingModel.model import BiSeNet

model = namedtuple("model", ["url", "model"])

models = {
    "face_detector": model(
        url="https://github.com/mike9251/simswap-inference-pytorch/releases/download/weights/face_detector_scrfd_10g_bnkps.onnx",
        model=FaceDetector,
    ),
    "arcface": model(
        url="https://github.com/mike9251/simswap-inference-pytorch/releases/download/weights/arcface_net.jit",
        model=FaceId,
    ),
    "generator_224": model(
        url="https://github.com/mike9251/simswap-inference-pytorch/releases/download/weights/simswap_224_latest_net_G.pth",
        model=Generator_Adain_Upsample,
    ),
    "generator_512": model(
        url="https://github.com/mike9251/simswap-inference-pytorch/releases/download/weights/simswap_512_390000_net_G.pth",
        model=Generator_Adain_Upsample,
    ),
    "parsing_model": model(
        url="https://github.com/mike9251/simswap-inference-pytorch/releases/download/weights/parsing_model_79999_iter.pth",
        model=BiSeNet,
    ),
}


def get_model(
    model_name: str,
    device: torch.device,
    load_state_dice: bool,
    model_path: Path,
    **kwargs,
):
    dst_dir = Path.cwd() / "weights"
    dst_dir.mkdir(exist_ok=True)

    url = models[model_name].url if not model_path.is_file() else str(model_path)

    if load_state_dice:
        model = models[model_name].model(**kwargs)

        if Path(url).is_file():
            state_dict = torch.load(url)
        else:
            state_dict = model_zoo.load_url(
                url,
                model_dir=str(dst_dir),
                progress=True,
                map_location="cpu",
            )

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
    else:
        dst_path = Path(url)

        if not dst_path.is_file():
            dst_path = dst_dir / Path(url).name

        if not dst_path.is_file():
            print(f"Downloading: '{url}' to {dst_path}")
            response = requests.get(url, stream=True)
            if int(response.status_code) == 200:
                file_size = int(response.headers["Content-Length"]) / (2**20)
                chunk_size = 1024
                bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n:3.1f}M/{total:3.1f}M [{elapsed}<{remaining}]"
                with open(dst_path, "wb") as handle:
                    with tqdm(total=file_size, bar_format=bar_format) as pbar:
                        for data in response.iter_content(chunk_size=chunk_size):
                            handle.write(data)
                            pbar.update(len(data) / (2**20))
            else:
                raise ValueError(
                    f"Couldn't download weights {url}. Specify weights for the '{model_name}' model manually."
                )

        kwargs.update({"model_path": str(dst_path), "device": device})
        model = models[model_name].model(**kwargs)

    return model
