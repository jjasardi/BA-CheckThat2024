from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import wandb
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns


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


def misclassified_samples(x_test, y_test, logits, model_names):
    for logits, model_name in zip(logits, model_names):
        predicted_labels = np.argmax(logits, axis=1)
        losses = compute_losses(logits, y_test)

        misclassified_indices = np.nonzero(predicted_labels != y_test)[0]
        misclassified_samples = [
            (x_test[index], predicted_labels[index], y_test[index], losses[index].item())
            for index in misclassified_indices
        ]
        misclassified_samples.sort(key=lambda x: x[3], reverse=True)

        print(model_name)
        for text, predicted_label, ground_truth_label, loss in misclassified_samples:
            print(
                f"{text}, Predicted Label = {predicted_label}, Ground Truth Label = {ground_truth_label}, loss = {loss}"
            )


def precision_recall_plot(y_test, logits, model_names):
    _, ax = plt.subplots()

    for logits, model_name in zip(logits, model_names):
        y_test = np.array(y_test, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, probs, name=model_name, ax=ax)

    plt.title('Precision-Recall curves')
    wandb.log({"Precision-Recall plot": wandb.Image(plt)})


def probability_calibration_plot(y_test, logits, model_names):
    _, ax = plt.subplots()
    for logits, model_name in zip(logits, model_names):
        y_test = np.array(y_test, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1]
        CalibrationDisplay.from_predictions(y_test, probs, name=model_name, ax=ax, n_bins=7)
    plt.title('Probability calibration curves')
    wandb.log({"Probability calibration plot": wandb.Image(plt)})

def models_outputs_correlation_matrix(logits):
    models_probs = [torch.nn.Softmax(dim=1)(torch.tensor(logit)) for logit in logits]
    correlation_matrix = np.zeros((len(models_probs), len(models_probs)))
    for i in range(len(models_probs)):
        for j in range(len(models_probs)):
            correlation_matrix[i, j] = pearsonr(models_probs[i][:, 1], models_probs[j][:, 1])[0]

    return correlation_matrix

def models_disagreement_matrix(logits):
    predicted_labels = [np.argmax(logit, axis=1) for logit in logits]
    num_models = len(predicted_labels)
    num_samples = predicted_labels[0].shape[0]

    disagreement_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(i + 1, num_models):
            disagreement_count = np.sum(predicted_labels[i] != predicted_labels[j])
            disagreement_matrix[i, j] = disagreement_count
            disagreement_matrix[j, i] = disagreement_count

    # Compute disagreement fraction
    disagreement_fraction_matrix = disagreement_matrix / num_samples

    return disagreement_fraction_matrix

def visualize_matrix(matrix, model_names, plot_name):
    df = pd.DataFrame(matrix, index=model_names, columns=model_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(plot_name)
    plt.xlabel('Model')
    plt.ylabel('Model')
    wandb.log({plot_name: wandb.Image(plt)})


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
    parser.add_argument("-l", "--logits", dest="logits_files", nargs="+", type=str, required=True)
    parser.add_argument("-n", "--model-names", dest="model_names", nargs="+", required=True)
    args = parser.parse_args()

    logits = []
    for logits_file in args.logits_files:
        logits.append(np.load("./model_dump/CT_24/" + logits_file))
    dataset = load(data_folder=args.data_folder, dev=args.dev_mode)
    x_test = [s.text for s in dataset.test]
    y_test = [s.class_label for s in dataset.test]

    misclassified_samples = misclassified_samples(x_test, y_test, logits, args.model_names)

    wandb.init(project="ba24-check-worthiness-estimation", group="general-plots", name="general-plots")

    precision_recall_plot(y_test, logits, args.model_names)

    # probability_calibration_plot(y_test, logits, args.model_names)

    correlation_matrix = models_outputs_correlation_matrix(logits)
    visualize_matrix(correlation_matrix, args.model_names, "Correlation of model predictions")
    
    disagreement_matrix = models_disagreement_matrix(logits)
    visualize_matrix(disagreement_matrix, args.model_names, "Disagreement of model predictions")
    