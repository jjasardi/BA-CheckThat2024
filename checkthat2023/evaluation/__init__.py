
from checkthat2023.evaluation.evaluation import evaluate, build_prediction_samples

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np


def hf_eval(eval_pred):
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    y_true = eval_pred.label_ids

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


