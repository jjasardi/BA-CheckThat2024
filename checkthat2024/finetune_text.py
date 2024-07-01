
from typing import List, Optional
from pathlib import Path
import datetime

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from checkthat2024.dataset_utils import TorchDataset

import numpy as np

from checkthat2024.task1a import Task1A
from checkthat2024.eval import hf_eval

import wandb
import os
os.environ["WANDB_LOG_MODEL"] = "end"


def finetune(
    dataset: Task1A,
    base_model: str,
    output_dir: Path,
    data_folder: str,
    dev_mode: bool = False,
):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"model_{current_time}"
    model_name = base_model.replace("/", "-")
    config = {"data": data_folder, "model": model_name}
    wandb.init(
        project="ba24-check-worthiness-estimation",
        name=f"{model_name}-{output_dir.name}",
        group=f"{model_name}-{data_folder}",
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

    x_train = [
        s.text
        for s in dataset.train
    ]
    y_train = [
        s.class_label
        for s in dataset.train
    ]
    x_dev = [
        s.text
        for s in dataset.dev
    ]
    y_dev = [
        s.class_label
        for s in dataset.dev
    ]
    x_test = [
        s.text
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
        logging_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
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

    print("EVALUATING MODEL")
    eval_result = trainer.evaluate(eval_dataset=dev)
    print(eval_result)

    print("GET PREDICTIONS")
    res = trainer.predict(
        test_dataset=test,
    )

    print("SCORE ON TEST DATASET")
    print(res.metrics)
    wandb.log({"test": res.metrics})

    print("SAVE PREDICTED LOGITS")
    with (output_dir / "text_model_test_logits.npy").open("wb") as fout:
        np.save(file=fout, arr=res.predictions)


if __name__ == "__main__":
    from checkthat2024.task1a import load
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
        data_folder=args.data_folder.name,
        dev_mode=args.dev_mode,
    )
