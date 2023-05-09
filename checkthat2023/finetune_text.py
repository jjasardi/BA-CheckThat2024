
from typing import List, Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

import torch
from torch.utils.data import Dataset as TDataset

from sklearn.metrics import f1_score

import numpy as np

from checkthat2023.tasks.task1a import Task1A


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


def compute_f1(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {
        "f1": f1_score(y_true=eval_pred.label_ids, y_pred=preds)
    }


def finetune(
    dataset: Task1A,
    base_model: str,
    output_dir: str,
    log_dir: str,
    dev_mode: bool = False,
):
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
        for s in dataset.dev_test
    ]
    y_test = [
        s.class_label
        for s in dataset.dev_test
    ]

    if dev_mode:
        x_train = x_train[:256]
        y_train = y_train[:256]
        x_dev = x_dev[:256]
        y_dev = y_dev[:256]
        x_test = x_test[:256]
        y_test = y_test[:256]

    train = TorchDataset.from_samples(x_train, y_train, tokenizer)
    dev = TorchDataset.from_samples(x_dev, y_dev, tokenizer)
    test = TorchDataset.from_samples(x_test, y_test, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if dev_mode else 10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=.01,
        learning_rate=5e-5,
        logging_dir=log_dir,
        logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        compute_metrics=compute_f1,
    )
    trainer.train()
    print("DONE TRAINING")
    res = trainer.predict(
        test_dataset=test,
    )
    print(res.metrics)

