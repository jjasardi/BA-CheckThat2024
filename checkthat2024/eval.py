from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import torch
from torch.nn import CrossEntropyLoss


def hf_eval(eval_pred):
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    y_true = eval_pred.label_ids

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average="binary",
    )
    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average="binary",
    )
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average="binary",
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_misclassified_samples(x_test, y_test, logits):
    predicted_labels = np.argmax(logits, axis=1)
    losses = compute_losses(logits, y_test)

    misclassified_indices = np.nonzero(predicted_labels != y_test)[0]
    misclassified_samples = [
        (x_test[index], predicted_labels[index], y_test[index], losses[index].item())
        for index in misclassified_indices
    ]
    misclassified_samples.sort(key=lambda x: x[3], reverse=True)
    return misclassified_samples


def compute_losses(logits, y_test):
    loss_func = CrossEntropyLoss(reduction="none")

    logits_tensor = torch.tensor(logits)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    losses = loss_func(logits_tensor, y_test_tensor)
    return losses


if __name__ == "__main__":
    from checkthat2024.task1a import load
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("--dev", dest="dev_mode", action="store_true")
    parser.add_argument("-l", "--logits", dest="logits_file", type=str, required=True)
    args = parser.parse_args()

    logits = np.load(args.logits_file)
    dataset = load(data_folder=args.data_folder, dev=args.dev_mode)
    x_test = [s.text for s in dataset.test]
    y_test = [s.class_label for s in dataset.test]

    misclassified_samples = get_misclassified_samples(x_test, y_test, logits)

    for text, predicted_label, ground_truth_label, loss in misclassified_samples:
        print(
            f"{text}, Predicted Label = {predicted_label}, Ground Truth Label = {ground_truth_label}, loss = {loss}"
        )
