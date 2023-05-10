
from pathlib import Path
from typing import List, Callable, Optional

from transformers import (
    AutoModel,
    AutoTokenizer,
    ViTModel,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from tqdm import tqdm

from checkthat2023.tasks.task1a import Task1A, Task1ASample
from checkthat2023.preprocessing.text import cardiffnlp_preprocess
from checkthat2023.evaluation import hf_eval


class TorchDataset(Dataset):

    def __init__(self, data: dict):
        self.data = data

    def __len__(self) -> int:
        return self.data['input_ids'].shape[0]

    def __getitem__(self, ix: int) -> dict:
        return {
            k: v[ix]
            for k, v in self.data.items()
        }

    @staticmethod
    def from_samples(
        samples: List[Task1ASample],
        tokenizer: AutoTokenizer,
        img_processor: ViTImageProcessor,
        txt_preprocess_fn: Callable[[str], str] = lambda s: s
    ) -> 'TorchDataset':
        result = tokenizer(
            [txt_preprocess_fn(s.tweet_text) for s in samples],
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        pix_vals = []
        for s in samples:
            with Image.open(s.image_path) as raw:
                if raw.mode != "RGB":
                    image = raw.convert('RGB')
                else:
                    image = raw
                pix_vals.append(
                    img_processor(image, return_tensors='pt')['pixel_values'])
        pix_vals = torch.cat(pix_vals, dim=0)

        result['pixel_values'] = pix_vals

        label_candidates = [s.class_label for s in samples]
        if all(l is not None for l in label_candidates):
            result['labels'] = torch.LongTensor([s.class_label for s in samples])

        return TorchDataset(result)


class MultiModalCrossAttention(nn.Module):

    def __init__(
        self,
        txt_model: AutoModel,
        img_model: ViTModel,
        shared_proj_dim: int = 256,
        finetune_base_models: bool = True,
    ):
        super().__init__()
        self.txt_model = txt_model
        self.img_model = img_model
        self.shared_proj_dim = shared_proj_dim
        self.finetune_base_models = finetune_base_models

        self.relu = nn.ReLU()

        self.txt2shared = nn.Linear(
            in_features=768,
            out_features=self.shared_proj_dim,
            bias=True,
        )
        self.img2shared = nn.Linear(
            in_features=768,
            out_features=self.shared_proj_dim,
            bias=True,
        )
        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=self.shared_proj_dim,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer=self.enc_layer,
            num_layers=1,
        )
        self.enc2logit = nn.Linear(
            in_features=self.shared_proj_dim,
            out_features=2,  # 2 classes
            bias=True,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def embedding(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        pixel_values: torch.tensor,
        **kwargs,
    ):
        with torch.set_grad_enabled(self.finetune_base_models):
            txt_out = self.txt_model(
                input_ids=input_ids, attention_mask=attention_mask)
            img_out = self.img_model(pixel_values=pixel_values)

        txt_proj = self.relu(self.txt2shared(txt_out.last_hidden_state))
        img_proj = self.relu(self.img2shared(img_out.last_hidden_state))

        shared = torch.cat([txt_proj, img_proj], dim=1)

        encoded = self.relu(self.enc(shared))

        mean_pooled = torch.mean(encoded, dim=1)

        embed = F.normalize(mean_pooled, dim=-1, p=2)

        return embed

    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        pixel_values: torch.tensor,
        labels: Optional[torch.LongTensor] = None,
    ):

        embed = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        logits = self.enc2logit(embed)

        if labels is None:
            loss = None
        else:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss,
        )


def build_kernel(
    train: TorchDataset,
    dev: TorchDataset,
    test: TorchDataset,
    model: MultiModalCrossAttention,
    output_dir: Path,
):
    model.cpu().eval()

    print("TRAIN EMBEDS")
    train_embeds = []
    with torch.no_grad():
        for ix in tqdm(range(len(train))):
            train_embeds.append(model.embedding(**train[ix:ix+1]))
    train_embeds = torch.cat(train_embeds, dim=0)

    print("DEV EMBEDS")
    dev_embeds = []
    with torch.no_grad():
        for ix in tqdm(range(len(dev))):
            dev_embeds.append(model.embedding(**dev[ix:ix+1]))
    dev_embeds = torch.cat(dev_embeds, dim=0)

    print("TEST EMBEDS")
    test_embeds = []
    with torch.no_grad():
        for ix in tqdm(range(len(test))):
            test_embeds.append(model.embedding(**test[ix:ix+1]))
    test_embeds = torch.cat(test_embeds, dim=0)

    with torch.no_grad():
        k_train = train_embeds @ train_embeds.T
        k_dev = dev_embeds @ train_embeds.T
        k_test = test_embeds @ train_embeds.T

    torch.save(obj=k_train, f=output_dir / "train_multimodal_sim.torch")
    torch.save(obj=k_dev, f=output_dir / "dev_multimodal_sim.torch")
    torch.save(obj=k_test, f=output_dir / "test_multimodal_sim.torch")


def finetune(
    dataset: Task1A,
    txt_model: str,
    img_model: str,
    output_dir: Path,
    finetune_base_models: bool = True,
    dev_mode: bool = False,
):
    if "cardiffnlp" in txt_model:
        txt_preprocess_fn = cardiffnlp_preprocess
    else:
        txt_preprocess_fn = lambda s: s

    tokenizer = AutoTokenizer.from_pretrained(txt_model)
    txt_model = AutoModel.from_pretrained(txt_model)

    img_processor = ViTImageProcessor.from_pretrained(img_model)
    img_model = ViTModel.from_pretrained(img_model)

    train = TorchDataset.from_samples(
        samples=dataset.train,
        tokenizer=tokenizer,
        img_processor=img_processor,
        txt_preprocess_fn=txt_preprocess_fn,
    )
    test = TorchDataset.from_samples(
        samples=dataset.test,
        tokenizer=tokenizer,
        img_processor=img_processor,
        txt_preprocess_fn=txt_preprocess_fn,
    )
    dev = TorchDataset.from_samples(
        samples=dataset.dev,
        tokenizer=tokenizer,
        img_processor=img_processor,
        txt_preprocess_fn=txt_preprocess_fn,
    )

    model = MultiModalCrossAttention(
        txt_model=txt_model,
        img_model=img_model,
        finetune_base_models=finetune_base_models,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if dev_mode else 10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=.01,
        learning_rate=5e-5,
        logging_dir=str(output_dir / "logs"),
        logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=hf_eval,
    )
    trainer.train()

    print("GET PREDICTIONS")
    res = trainer.predict(
        test_dataset=test,
    )
    print("SAVE PREDICTED LOGITS")
    with (output_dir / "multimodal_test_logits.npy").open("wb") as fout:
        np.save(file=fout, arr=res.predictions)

    print("CREATE KERNEL")
    build_kernel(
        train=train,
        dev=dev,
        test=test,
        model=model,
        output_dir=output_dir,
    )
