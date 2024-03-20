
from typing import List, Optional
from pathlib import Path
import datetime

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

import torch
from torch.utils.data import Dataset as TDataset

import numpy as np

from checkthat2023.tasks.task1a import Task1A
from checkthat2023.evaluation import hf_eval


class TorchDataset(TDataset):

    def __init__(
        self,
        torch_data: dict
    ):
        self.torch_data = torch_data

    def __len__(self) -> int:
        return self.torch_data['input_ids'].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            k: v[idx]
            for k, v in self.torch_data.items()
        }

    @staticmethod
    def from_samples(
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer: AutoTokenizer,
    ) -> 'TorchDataset':
        torch_data = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt")
        if labels is not None:
            torch_data['labels'] = torch.LongTensor(labels)
        return TorchDataset(torch_data)


def finetune(
    dataset: Task1A,
    base_model: str,
    output_dir: Path,
    dev_mode: bool = False,
):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"model_{current_time}"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

    x_train = [
        s.tweet_text
        for s in dataset.train
    ]
    y_train = [
        s.class_label
        for s in dataset.train
    ]
    x_dev = [
        s.tweet_text
        for s in dataset.dev
    ]
    y_dev = [
        s.class_label
        for s in dataset.dev
    ]
    x_test = [
        s.tweet_text
        for s in dataset.test
    ]
    y_test = [
        s.class_label if hasattr(s, 'class_label') else None
        for s in dataset.test
    ]

    train = TorchDataset.from_samples(x_train, y_train, tokenizer)
    dev = TorchDataset.from_samples(x_dev, y_dev, tokenizer)
    test = TorchDataset.from_samples(x_test, y_test, tokenizer)

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
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=hf_eval,
    )
    trainer.train()

    print("SAVING MODEL")
    trainer.save_model(output_dir=str(output_dir / "text_model"))
    tokenizer.save_pretrained(save_directory=str(output_dir / "text_model"))

    print("GET PREDICTIONS")
    res = trainer.predict(
        test_dataset=test,
    )

    print("SCORE ON TEST DATASET")
    print(res.metrics)

    print("SAVE PREDICTED LOGITS")
    with (output_dir / "text_model_test_logits.npy").open("wb") as fout:
        np.save(file=fout, arr=res.predictions)


if __name__ == "__main__":
    from checkthat2023.tasks.task1a import load
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("-m", "--base-model", dest="base_model", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--dev", dest="dev_mode", action="store_true")

    args = parser.parse_args()

    finetune(
        dataset=load(data_folder=args.data_folder, dev=args.dev_mode),
        base_model=args.base_model,
        output_dir=args.output_dir,
        dev_mode=args.dev_mode,
    )
