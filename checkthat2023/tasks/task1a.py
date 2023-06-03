
from typing import List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from checkthat2023.tasks.base import Sample


@dataclass(frozen=True)
class Task1ASample(Sample):
    tweet_url: str
    tweet_text: str
    ocr_text: str
    image_path: Path
    image_url: str

    @staticmethod
    def from_sample_dict(
        tweet_id: str,
        tweet_url: str,
        tweet_text: str,
        ocr_text: str,
        image_path: str,
        image_url: str,
        data_folder: Path,
        class_label: Optional[str] = None,
    ) -> 'Task1ASample':

        if class_label is not None and class_label not in {'Yes', 'No'}:
            raise ValueError(f"expect 'class_label' to be one of ['Yes', 'No'],"
                             f"received '{class_label}'")

        return Task1ASample(
            id=tweet_id,
            tweet_url=tweet_url,
            tweet_text=tweet_text,
            ocr_text=ocr_text,
            class_label=class_label == 'Yes' if class_label is not None else None,
            image_path=data_folder / "task1A" / image_path,
            image_url=image_url,
        )


@dataclass(frozen=True)
class Task1A:
    train: List[Task1ASample]
    dev: List[Task1ASample]
    test: List[Task1ASample]


def load(data_folder: Path, dev: bool = False) -> Task1A:
    args = {}

    for split in ["train", "dev", "dev_test", "test"]:
        data_file = data_folder / "task1A" /\
                    f"CT23_1A_checkworthy_multimodal_english_{split}.jsonl"
        if split == "test":
            data_file = data_folder / "task1A" /\
                    "CT23_1A_checkworthy_multimodal_english_test_gold.jsonl"
        with data_file.open('r') as fin:
            raw = [json.loads(line.strip()) for line in fin]

        args[split] = [
            Task1ASample.from_sample_dict(**d, data_folder=data_folder)
            for d in raw
        ]

    if dev:
        args = {
            k: v[:10]
            for k, v in args.items()
        }

    return Task1A(
        train=args['train'],
        dev=args['dev'] + args['dev_test'],
        test=args['test'],
    )
