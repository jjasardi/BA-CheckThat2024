
from typing import Literal, List
from pathlib import Path
from dataclasses import dataclass, asdict

import csv


@dataclass(frozen=True)
class SubmissionEntry:
    tweet_id: int
    class_label: Literal['Yes', 'No']
    run_id: str


def create_submission(
    sub_data: List[SubmissionEntry],
    sub_file: Path,
):
    with sub_file.open('w') as fout:
        fieldnames = ['tweet_id', 'class_label', 'run_id']
        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
            delimiter='\t',
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writerows(map(asdict, sub_data))
