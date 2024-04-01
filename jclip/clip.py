import hashlib
import os
import numpy as np
import urllib
import warnings
from typing import Union, List

import jittor as jt
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from PIL import Image
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50":
    "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101":
    "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4":
    "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16":
    "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64":
    "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32":
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16":
    "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14":
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px":
    "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target,
                               "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target,
                                                     "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")),
                  ncols=80,
                  unit='iB',
                  unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target,
                           "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def to_tensor(data):
    return jt.Var(data)


class ImageToTensor(object):

    def __call__(self, input):
        input = np.asarray(input)
        if len(input.shape) < 3:
            input = np.expand_dims(input, -1)
        return to_tensor(input)


class Resize:

    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            self.size = _setup_size(
                size,
                error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if isinstance(self.size, int):
            w, h = img.size

            short, long = (w, h) if w <= h else (h, w)
            if short == self.size:
                return img

            new_short, new_long = self.size, int(self.size * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                                 new_short)
            size = (new_h, new_w)
        return resize(img, size, self.mode)


def _transform(n_px):
    return Compose([
        Resize(n_px, mode=Image.BICUBIC),
        CenterCrop(n_px), _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name, download_root=None):
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root
            or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}")

    # with open(model_path, 'rb') as opened_file:
    state_dict = jt.load(model_path)

    model = build_model(state_dict)
    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]

    result = jt.zeros((len(all_tokens), context_length), dtype=jt.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = jt.Var(tokens)

    return result
