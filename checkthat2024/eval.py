from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from scipy import stats
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
import random
from itertools import combinations

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from checkthat2024.dataset_utils import TorchDataset
from checkthat2024.calibration import PlattScaling
from checkthat2024.calibration import Isotonic

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


def probability_calibration_plot(y_test, logits, model_labels, fitted_calibration):
    _, ax = plt.subplots()
    for logits, model_label in zip(logits, model_labels):
        y_test = np.array(y_test, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1]
        probs = fitted_calibration.transform(probs)
        CalibrationDisplay.from_predictions(y_test, probs, name=model_label, ax=ax, n_bins=5)
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
    m_pairwise_disagreement = mean_pairwise_disagreement(disagreement_fraction_matrix, list(range(num_models)))
    wandb.log({"mean_pairwise_disagreement": m_pairwise_disagreement})
    return disagreement_fraction_matrix

def models_normalized_disagreement_matrix(logits, y_test):
    predicted_labels = [np.argmax(logit, axis=1) for logit in logits]
    num_models = len(predicted_labels)
    num_samples = predicted_labels[0].shape[0]

    disagreement_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(i + 1, num_models):
            disagreement_count = np.sum(predicted_labels[i] != predicted_labels[j])
            disagreement_fraction = disagreement_count / num_samples
            majority_vote = stats.mode([predicted_labels[i], predicted_labels[j]], keepdims=False)[0]
            accuracy = accuracy_score(y_test, majority_vote)
            normalized_disagreement = disagreement_fraction / (1 - accuracy)

            disagreement_matrix[i, j] = normalized_disagreement
            disagreement_matrix[j, i] = normalized_disagreement

    m_pairwise_disagreement = mean_pairwise_disagreement(disagreement_matrix, list(range(num_models)))
    wandb.log({"mean_pairwise_normalized_disagreement": m_pairwise_disagreement})
    return disagreement_matrix


def double_fault_measure_matrix(logits, y_test):
    predicted_labels = [np.argmax(logit, axis=1) for logit in logits]
    num_models = len(predicted_labels)
    num_samples = predicted_labels[0].shape[0]

    double_fault_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(i + 1, num_models):
            both_incorrect = np.logical_and(predicted_labels[i] != y_test, predicted_labels[j] != y_test)
            double_fault_fraction = np.sum(both_incorrect) / num_samples

            double_fault_matrix[i, j] = double_fault_fraction
            double_fault_matrix[j, i] = double_fault_fraction

    m_pairwise_double_fault = mean_pairwise_disagreement(double_fault_matrix, list(range(num_models)))
    wandb.log({"mean_pairwise_double_fault_measure": m_pairwise_double_fault})
    return double_fault_matrix


def mean_pairwise_disagreement(disagreement_matrix, model_indices):
    """
    Calculate the mean pairwise disagreement for a given set of model indices.

    :param disagreement_matrix: A 2D numpy array representing the disagreement fractions between models.
    :param model_indices: A list of indices representing the models in the group.
    :return: The mean pairwise disagreement for the given set of models.
    """
    pairwise_disagreements = []
    for (i, j) in combinations(model_indices, 2):
        pairwise_disagreements.append(disagreement_matrix[i, j])
    m_pairwise_disagreement = np.mean(pairwise_disagreements)
    return m_pairwise_disagreement

def compute_metrics(logits, y_test, model_combination):
    # Extract the predictions for the models in the combination
    all_predictions = [logits[model_idx] for model_idx in model_combination]
    # Get the predicted labels by majority voting
    all_predicted_labels = [np.argmax(prediction, axis=1) for prediction in all_predictions]
    majority_vote = stats.mode(all_predicted_labels, axis=0, keepdims=False)[0]
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, majority_vote)
    precision = precision_score(y_test, majority_vote)
    recall = recall_score(y_test, majority_vote)
    f1 = f1_score(y_test, majority_vote)
    return accuracy, precision, recall, f1


def get_all_combinations_with_disagreement(disagreement_matrix, logits, y_test):
    """
    Generate a list of all possible combinations of models with their mean pairwise disagreement,
    and compute the evaluation metrics for the top 10 and last 10 combinations.

    :param disagreement_matrix: A 2D numpy array representing the disagreement fractions between models.
    :param logits: A list of numpy arrays representing the logits from each model.
    :param y_test: A numpy array representing the true labels.
    :return: A list of tuples, each containing a combination of models, their mean pairwise disagreement,
             and their evaluation metrics (accuracy, precision, recall, F1 score).
    """
    num_models = disagreement_matrix.shape[0]
    all_combinations = []

    for r in range(3, num_models + 1, 2):  # Only consider combinations with an odd number of models
        for model_combination in combinations(range(num_models), r):
            mean_disagreement = mean_pairwise_disagreement(disagreement_matrix, model_combination)
            accuracy, precision, recall, f1 = compute_metrics(logits, y_test, model_combination)
            all_combinations.append((model_combination, mean_disagreement, accuracy, precision, recall, f1))

    #Sort the combinations by mean pairwise disagreement in descending order
    all_combinations.sort(key=lambda x: x[1], reverse=True)

    #top_10 = all_combinations[:10]
    #bottom_10 = all_combinations[-10:]

    # Compute evaluation metrics for top 10 and bottom 10 combinations
    #results = []
    #for combination in top_10 + bottom_10:
    #    model_combination, mean_disagreement = combination
    #    accuracy, precision, recall, f1 = compute_metrics(logits, y_test, model_combination)
    #    results.append((model_combination, mean_disagreement, accuracy, precision, recall, f1))

    return all_combinations

def f1_for_thresholds(logits, y, model_labels, data_label, fitted_calibration):
    for logits, model_label in zip(logits, model_labels):
        y = np.array(y, dtype=int)
        logits_tensor = torch.tensor(logits)
        probs = torch.nn.Softmax(dim=1)(logits_tensor)[:, 1].numpy()

        calibrated_probs = fitted_calibration.transform(probs)

        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        thresholds = np.percentile(calibrated_probs, percentiles)

        print(f"Model: {model_label}, data: {data_label}")
        table = wandb.Table(columns=["Threshold", "F1_score"])
        for threshold in thresholds:
            predicted_labels = (calibrated_probs >= threshold).astype(int)
            f1 = f1_score(y, predicted_labels)
            print(f"Threshold: {threshold:.8f}, F1 Score: {f1}")
            table.add_data(threshold, f1)
        #wandb.log({f"{model_label}_{data_label}": table})

def get_fitted_calibration_method(dataset, model_name):
    random_indices = random.sample(range(len(dataset.train)), 100)
    x_train = []
    y_train = []
    for index in random_indices:
        x_train.append(dataset.train[index].text)
        y_train.append(dataset.train[index].class_label)

    predictions = get_predictions(model_name, x_train, y_train)
    predictions = torch.tensor(predictions)
    probs_yes_class = torch.nn.Softmax(dim=1)(predictions)[:, 1].numpy()

    calibration_method = Isotonic()
    calibration_method.fit(probs_yes_class, y_train)
    return calibration_method

def get_predictions(model_name, x, y):
    model = AutoModelForSequenceClassification.from_pretrained("./model_dump/CT_24/" + model_name + "/text_model")
    tokenizer = AutoTokenizer.from_pretrained("./model_dump/CT_24/" + model_name + "/text_model")
    test = TorchDataset.from_samples(x, y, tokenizer)
    trainer = Trainer(model=model)
    predictions, _, metrics = trainer.predict(test_dataset=test)
    print(metrics)
    return predictions

def visualize_matrix(matrix, model_labels, plot_name):
    df = pd.DataFrame(matrix, index=model_labels, columns=model_labels)
    print(matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".3f")
    plt.title(plot_name)
    plt.xlabel('Model')
    plt.ylabel('Model')
    plt.tight_layout()
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

    #model_names = []
    #for file in args.logits_files:
    #    model_name = os.path.basename(os.path.dirname(file))
    #    model_names.append(model_name)
    #predictions = get_predictions(model_names[0], x_test, y_test)
    #print(f1_score(y_test, np.argmax(predictions, axis=1)))

    misclassified_samples = misclassified_samples(x_test, y_test, logits, args.model_labels)

    wandb.init(project="ba24-check-worthiness-estimation", mode="open", group="general-plots", name="general-plots", config={"data":args.data_folder.name})

    precision_recall_plot(y_test, logits, args.model_labels)

    if len(args.logits_files) > 2:
        correlation_matrix = models_outputs_correlation_matrix(logits)
        visualize_matrix(correlation_matrix, args.model_labels, "Correlation of model predictions")

        disagreement_matrix = models_disagreement_matrix(logits)
        visualize_matrix(disagreement_matrix, args.model_labels, "Disagreement of model predictions")

    model_names = []
    for file in args.logits_files:
        model_name = os.path.basename(os.path.dirname(file))
        model_names.append(model_name)

    fitted_calibration = get_fitted_calibration_method(dataset, model_name)

    x_dev = [s.text for s in dataset.dev]
    y_dev = [s.class_label for s in dataset.dev]
    logits_dev = []
    for model_name in model_names:
        logits_dev.append(get_predictions(model_name, x_dev, y_dev))
    f1_for_thresholds(logits_dev, y_dev, args.model_labels, "y_dev", fitted_calibration)

    f1_for_thresholds(logits, y_test, args.model_labels, "y_test", fitted_calibration)

    probability_calibration_plot(y_test, logits, args.model_labels, fitted_calibration)

