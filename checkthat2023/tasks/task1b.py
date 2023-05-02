
from typing import List
from dataclasses import dataclass
import csv
from pathlib import Path

from checkthat2023.tasks.base import Sample


@dataclass(frozen=True)
class Task1BSample(Sample):
    text: str

    @staticmethod
    def from_sample_dict(
        sentence_id: str,
        text: str,
        class_label: str,
    ) -> 'Task1BSample':

        if class_label not in {'Yes', 'No'}:
            raise ValueError(f"expect 'class_label' to be one of ['Yes', 'No'],"
                             f"received '{class_label}'")

        return Task1BSample(
            id=sentence_id,
            text=text,
            class_label=class_label == 'Yes',
        )


@dataclass(frozen=True)
class Task1B:
    train: List[Task1BSample]
    dev: List[Task1BSample]
    dev_test: List[Task1BSample]


def load(data_folder: Path, dev: bool = False) -> Task1B:
    args = {}

    for split in ["train", "dev", "dev_test"]:
        fname = f"CT23_1{'B' if split == 'dev_test' else 'C'}_checkworthy_english_{split}.tsv"

        with (data_folder / "task1B" / fname).open('r') as fin:
            reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            raw = list(reader)

        args[split] = [
            Task1BSample.from_sample_dict(**{
                k.lower(): v
                for k, v in d.items()
            })
            for d in raw
        ]

    if dev:
        args = {
            k: v[:10]
            for k, v in args.items()
        }

    return Task1B(**args)
