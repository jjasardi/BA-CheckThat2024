
from typing import List
from dataclasses import dataclass
import json

from PIL import Image

from datasets.util import task1a_folder


@dataclass(frozen=True)
class Task1ASample:
    id: str
    tweet_url: str
    tweet_text: str
    ocr_text: str
    class_label: bool
    image_path: str
    image_url: str
    image: Image

    @staticmethod
    def from_sample_dict(
        tweet_id: str,
        tweet_url: str,
        tweet_text: str,
        ocr_text: str,
        class_label: str,
        image_path: str,
        image_url: str,
    ) -> 'Task1ASample':

        if class_label not in {'Yes', 'No'}:
            raise ValueError(f"expect 'class_label' to be one of ['Yes', 'No'],"
                             f"received '{class_label}'")

        with Image.open(task1a_folder() / image_path) as img:
            image = img.load()
        return Task1ASample(
            id=tweet_id,
            tweet_url=tweet_url,
            tweet_text=tweet_text,
            ocr_text=ocr_text,
            class_label=class_label == 'Yes',
            image_path=image_path,
            image_url=image_url,
            image=image,
        )


@dataclass(frozen=True)
class Task1A:
    train: List[Task1ASample]
    dev: List[Task1ASample]
    dev_test: List[Task1ASample]


def load() -> Task1A:
    args = {}

    for split in ["train", "dev", "dev_test"]:
        data_file = task1a_folder() /\
                    f"CT23_1A_checkworthy_multimodal_english_{split}.jsonl"
        with data_file.open('r') as fin:
            raw = [json.loads(line.strip()) for line in fin]

        args[split] = [
            Task1ASample.from_sample_dict(**d)
            for d in raw
        ]

    return Task1A(**args)
