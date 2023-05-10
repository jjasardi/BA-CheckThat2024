
from typing import List
from pathlib import Path

from transformers import (
    ViTImageProcessor,
    ViTModel,
)

import torch
import torch.nn.functional as F

from PIL import Image

from tqdm import tqdm

from checkthat2023.tasks.task1a import Task1ASample, Task1A


class Embedder:

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k"):
        self.model_name = model_name
        self.img_proc = ViTImageProcessor.from_pretrained(model_name)
        self.mod = ViTModel.from_pretrained(model_name)

    def embeddings(self, samples: List[Task1ASample]):
        res = []
        for s in tqdm(samples):
            with Image.open(s.image_path) as img:
                feats = self.img_proc(
                    img if img.mode == 'RGB' else img.convert('RGB'),
                    return_tensors='pt',
                )
            with torch.no_grad():
                out = self.mod(**feats)

            res.append(out.pooler_output)

        res = torch.cat(res)

        with torch.no_grad():
            res = F.normalize(res, dim=-1, p=2)

        return res


def img_kernel_untrained(
    dataset: Task1A,
    output: Path,
):
    embedder = Embedder()

    print("TRAIN EMBEDS")
    train = embedder.embeddings(dataset.train)

    print("DEV EMBEDS")
    dev = embedder.embeddings(dataset.dev)

    print("TEST EMBEDS")
    test = embedder.embeddings(dataset.test)

    k_train = train @ train.T
    torch.save(k_train, output / "train_img_untrained_sim.torch")

    k_dev = dev @ train.T
    torch.save(k_dev, output / "dev_img_untrained_sim.torch")

    k_test = test @ train.T
    torch.save(k_test, output / "test_img_untrained_sim.torch")
