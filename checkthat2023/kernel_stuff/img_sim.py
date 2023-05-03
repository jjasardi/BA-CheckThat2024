
from typing import List, Optional
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from transformers import ViTImageProcessor, ViTModel

from tqdm import tqdm


class ImageSimilarityKernel:

    def __init__(self):
        self.model_name = "google/vit-base-patch16-224-in21k"
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name)

    @staticmethod
    def __preprocess(img: Image) -> Image:
        if img.mode != "RGB":
            return img.convert('RGB')
        else:
            return img

    def img_embed(self, img_path: Path):
        raw = Image.open(img_path)

        img = self.__preprocess(raw)

        features = self.feature_extractor(img, return_tensors='pt')
        with torch.no_grad():
            out = self.model(**features)

        raw.close()

        return F.normalize(out.pooler_output[0], dim=-1, p=2)

    def kernel(self, x: List[Path], y: Optional[List[Path]] = None):
        x_embeds = []
        for p in tqdm(x):
            x_embeds.append(self.img_embed(p))
        x_embeds = torch.stack(x_embeds)

        if y is None:
            y_embeds = x_embeds
        else:
            y_embeds = []
            for p in tqdm(y):
                y_embeds.append(self.img_embed(p))
            y_embeds = torch.stack(y_embeds)

        return torch.einsum("ik,jk -> ij", x_embeds, y_embeds)
