
from typing import List, Optional
from dataclasses import dataclass
import csv
from pathlib import Path

from checkthat2024.base import Sample


@dataclass(frozen=True)
class Task1ASample(Sample):
    text: str
    
    @staticmethod
    def from_sample_dict(
        Sentence_id: str,
        Text: str,
        class_label: Optional[str] = None,
    ) -> 'Task1ASample':

        if class_label is not None and class_label not in {'Yes', 'No'}:
            raise ValueError(f"expect 'class_label' to be one of ['Yes', 'No'],"
                             f"received '{class_label}'")

        return Task1ASample(
            id=Sentence_id,
            text=Text,
            class_label=class_label == 'Yes' if class_label is not None else None,
        )


@dataclass(frozen=True)
class Task1A:
    train: List[Task1ASample]
    dev: List[Task1ASample]
    test: List[Task1ASample]
    test_gold: List[Task1ASample]


def load(data_folder: Path, dev: bool = False, gold: bool = False) -> Task1A:
    args = {}

    splits = ["train", "dev", "dev-test"]
    if gold:
        splits.append("test_gold")

    for split in splits:
        data_file = data_folder / "CT24_checkworthy_english" /\
                    f"CT24_checkworthy_english_{split}.tsv"
        with data_file.open('r', encoding='utf-8') as fin:
            raw = list(csv.DictReader(fin, delimiter='\t'))

        args[split] = [
            Task1ASample.from_sample_dict(**d)
            for d in raw
        ]

    if dev:
        args = {
            k: v[:10]
            for k, v in args.items()
        }

    return Task1A(
        train=args['train'],
        dev=args['dev'],
        test=args['dev-test'],
        test_gold=args['test_gold'] if gold else None
    )
