
from pathlib import Path

from checkthat2023.preprocessing.text import cardiffnlp_preprocess
from checkthat2023.tasks.task1a import load

from transformers import (
    AutoModel,
    AutoTokenizer,
    ViTImageProcessor,
    ViTModel,
)

import torch

from PIL import Image


data_path = Path('data')

task1a = load(data_folder=data_path, dev=True)

txt_model = "cardiffnlp/twitter-roberta-base"
img_model = "google/vit-base-patch16-224-in21k"

tok = AutoTokenizer.from_pretrained(txt_model)
img_proc = ViTImageProcessor.from_pretrained(img_model)

from checkthat2023.finetune_multi import TorchDataset

train = TorchDataset.from_samples(
    samples=task1a.train,
    tokenizer=tok,
    img_processor=img_proc,
)
