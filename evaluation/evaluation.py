
from typing import List
from dataclasses import dataclass

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)

from tasks.base import Sample


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def __repr__(self) -> str:
        return f"acc: {self.accuracy:.4f}, P: {self.precision:.4f}," \
               f" R: {self.recall:.4f}, F1: {self.f1:.4f}"


def build_prediction_samples(
    gold_samples: List[Sample],
    raw_predictions: List[bool],
) -> List[Sample]:
    return [
        Sample(
            id=gold.id,
            class_label=raw_label,
        )
        for gold, raw_label in zip(gold_samples, raw_predictions)
    ]


def evaluate(gold: List[Sample], prediction: List[Sample]):
    assert len(gold) == len(prediction)
    for g, p in zip(gold, prediction):
        assert g.id == p.id

    y_true = [g.class_label for g in gold]
    y_pred = [p.class_label for p in prediction]

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=True,
        average='binary',
    )
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=True,
        average='binary',
    )
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=True,
        average='binary',
    )

    return EvalResult(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
    )


