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

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from dataset_utils import TorchDataset

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


def misclassified_samples(x_test, y_test, logits, model_labels):
    for logits, model_label in zip(logits, model_labels):
        predicted_labels = np.argmax(logits, axis=1)
        losses = compute_losses(logits, y_test)

        misclassified_indices = np.nonzero(predicted_labels != y_test)[0]
        misclassified_samples = [
            (x_test[index], predicted_labels[index], y_test[index], losses[index].item())
            for index in misclassified_indices
        ]
        misclassified_samples.sort(key=lambda x: x[3], reverse=True)

        print(model_label)
        for text, predicted_label, ground_truth_label, loss in misclassified_samples:
            print(
                f"{text}, Predicted Label = {predicted_label}, Ground Truth Label = {ground_truth_label}, loss = {loss}"
            )


def precision_recall_plot(y_test, logits, model_labels):
    _, ax = plt.subplots()

    for logits, model_label in zip(logits, model_labels):
        y_test = np.array(y_test, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, probs, name=model_label, ax=ax)

    plt.title('Precision-Recall curves')
    wandb.log({"Precision-Recall plot": wandb.Image(plt)})


def probability_calibration_plot(y_test, logits, model_labels):
    _, ax = plt.subplots()
    for logits, model_label in zip(logits, model_labels):
        y_test = np.array(y_test, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1]
        CalibrationDisplay.from_predictions(y_test, probs, name=model_label, ax=ax, n_bins=7)
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

def mean_pairwise_disagreement(disagreement_fraction_matrix):
    num_models = disagreement_fraction_matrix.shape[0]
    total_disagreement = np.sum(disagreement_fraction_matrix)
    # Subtract diagonal elements (self-disagreements) from total
    total_disagreement -= np.sum(np.diag(disagreement_fraction_matrix))
    # Divide by number of pairs (combinations of 2 models)
    mean_disagreement = total_disagreement / (num_models * (num_models - 1))
    return mean_disagreement

def f1_for_thresholds(logits, y, model_labels, data_label):
    thresholds = np.arange(0.05, 1, 0.05)
    for logits, model_label in zip(logits, model_labels):
        y = np.array(y, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1].numpy()

        print(f"Model: {model_label}, data: {data_label}")
        table = wandb.Table(columns=["Threshold", "F1_score"])
        for threshold in thresholds:
            predicted_labels = (probs >= threshold).astype(int)
            f1 = f1_score(y, predicted_labels)
            print(f"Threshold: {threshold:.2f}, F1 Score: {f1:.4f}")
            table.add_data(threshold, f1)
        wandb.log({f"{model_label}_{data_label}": table})

def get_predictions(model_name):
    model = AutoModelForSequenceClassification.from_pretrained("./model_dump/CT_24/" + model_name + "/text_model")
    tokenizer = AutoTokenizer.from_pretrained("./model_dump/CT_24/" + model_name + "/text_model")

    test = TorchDataset.from_samples(x_test, y_test, tokenizer)
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset=test).predictions
    return predictions

def visualize_matrix(matrix, model_labels, plot_name):
    df = pd.DataFrame(matrix, index=model_labels, columns=model_labels)

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
    import os

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("--dev", dest="dev_mode", action="store_true")
    parser.add_argument("-l", "--logits", dest="logits_files", nargs="+", type=str, required=True)
    parser.add_argument("-n", "--model-labels", dest="model_labels", nargs="+", required=True)
    args = parser.parse_args()

    logits = []
    for logits_file in args.logits_files:
        logits.append(np.load("./model_dump/CT_24/" + logits_file))
    dataset = load(data_folder=args.data_folder, dev=args.dev_mode)
    x_test = [s.text for s in dataset.test]
    y_test = [s.class_label for s in dataset.test]

    misclassified_samples = misclassified_samples(x_test, y_test, logits, args.model_labels)

    wandb.init(project="ba24-check-worthiness-estimation", group="general-plots", name="general-plots", config={"data":args.data_folder.name})

    precision_recall_plot(y_test, logits, args.model_labels)

    # probability_calibration_plot(y_test, logits, args.model_labels)

    correlation_matrix = models_outputs_correlation_matrix(logits)
    visualize_matrix(correlation_matrix, args.model_labels, "Correlation of model predictions")

    disagreement_matrix = models_disagreement_matrix(logits)
    visualize_matrix(disagreement_matrix, args.model_labels, "Disagreement of model predictions")

    model_names = []
    for file in args.logits_files:
        model_name = os.path.basename(os.path.dirname(file))
        model_names.append(model_name)


    y_dev = [s.class_label for s in dataset.dev]
    f1_for_thresholds(logits, y_test, args.model_labels, "y_test")
    logits_dev = []
    for model_name in model_names:
        logits_dev.append(get_predictions(model_name))
    f1_for_thresholds(logits_dev, y_dev, args.model_labels, "y_dev")