from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from checkthat2024.dataset_utils import TorchDataset

from checkthat2024.task1a import Task1A, load

from pathlib import Path
import numpy as np
from scipy import stats
from checkthat2024.sample_length import process_texts
from checkthat2024.eval import (get_predictions, get_all_combinations_with_disagreement)

import wandb

def ensemble(
    dataset: Task1A,
    model_names: list[str],
    processed_models: list[int]=[],
    voting: str='hard',
):
    x_test = [
            s.text
            for s in dataset.test_gold
        ]
    y_test = [
        s.class_label if hasattr(s, 'class_label') else None
        for s in dataset.test_gold
    ]

    all_predictions = []
    for i, model_name in enumerate(model_names):     
        test_texts = x_test
        if i in processed_models:
            test_texts = process_texts(x_test)

        model_predictions = get_predictions(model_name, test_texts, y_test)
        print(f1_score(y_test, np.argmax(model_predictions, axis=1)))
        all_predictions.append(model_predictions)
    
    
    if voting =='hard':
        all_predicted_labels = [np.argmax(prediction, axis=1) for prediction in all_predictions]
        majority_vote = stats.mode(all_predicted_labels, keepdims=False)[0]
        
        print(all_predicted_labels)
        print(majority_vote)
        
        metrics = {
        "test_f1": f1_score(y_test, majority_vote),
        "test_recall": recall_score(y_test, majority_vote),
        "test_precision": precision_score(y_test, majority_vote),
        "test_accuracy": accuracy_score(y_test, majority_vote),
        }
        wandb.log({"test": metrics})

    potential = 0
    for i in range(len(y_test)):
        for prediction in all_predictions:
            if np.argmax(prediction[i]) == y_test[i]:
                potential += 1
                break
    wandb.log({"potential": potential})
    return majority_vote



if __name__ == "__main__":
    from checkthat2024.task1a import load
    from argparse import ArgumentParser
    import sys
    from checkthat2024.eval import models_disagreement_matrix, visualize_matrix, models_normalized_disagreement_matrix, double_fault_measure_matrix

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("-l", "--models", dest="model_names", nargs="+", type=str, required=True)
    parser.add_argument("-v", "--voting", dest="voting", type=str, required=False)
    parser.add_argument("-n", "--model-labels", dest="model_labels", nargs="+", required=False)
    parser.add_argument("-p", "--processed-models", dest="processed_models", nargs="+", type=int, required=False, default=[])

    args = parser.parse_args()
    if len(args.model_names) < 2:
        print("please provide more than one model")
        sys.exit()

    model = "ensemble"
    config = {"data": args.data_folder, "model": model, "voting": args.voting}

    wandb.init(
        project="ba24-check-worthiness-estimation",
        name=f"ensemble-{'-'.join(args.model_names)}",
        group=f"ensemble-{args.data_folder}",
        config=config,
    )
    dataset = load(data_folder=args.data_folder, gold=True)
    ensemble(
        dataset=dataset,
        model_names=args.model_names,
        voting=args.voting,
        processed_models=args.processed_models,
    )

    y_test = [
        s.class_label if hasattr(s, 'class_label') else None
        for s in dataset.test_gold
    ]

    if len(args.model_labels) == len(args.model_names):
        logits = []
        for model in args.model_names:
            logits.append(np.load("./model_dump/CT_24/" + model + "/text_model_test_logits.npy"))
        disagreement_matrix = models_disagreement_matrix(logits)
        visualize_matrix(disagreement_matrix, args.model_labels, "Disagreement of models")
        normalized_disagreement_matrix = models_normalized_disagreement_matrix(logits, y_test)
        visualize_matrix(normalized_disagreement_matrix, args.model_labels, "Normalized disagreement of models")
        double_fault_matrix = double_fault_measure_matrix(logits, y_test)
        visualize_matrix(double_fault_matrix, args.model_labels, "Double fault measure of models")

        all_combinations_dis = get_all_combinations_with_disagreement(disagreement_matrix, logits, y_test)
        for combination, mean_disagreement, acc, prec, rec, f1 in all_combinations_dis:
            print(f"Models: {combination}, Mean Pairwise Disagreement: {mean_disagreement:.4f}, metrics: {acc:.4f}, {prec:.4f}, {rec:.4f}, {f1:.4f}")

        all_combinations_df = get_all_combinations_with_disagreement(double_fault_matrix, logits, y_test)
        for combination, mean_double_fault, acc, prec, rec, f1 in all_combinations_df:
            print(f"Models: {combination}, Mean double fault measure: {mean_double_fault:.4f}, metrics: {acc:.4f}, {prec:.4f}, {rec:.4f}, {f1:.4f}")

        # Print top 10 and bottom 10 combinations for disagreement matrix
        print("Top 10 combinations based on mean pairwise disagreement:")
        for combination, mean_disagreement, acc, prec, rec, f1 in all_combinations_dis[:10]:
            print(f"Models: {combination}, Mean Pairwise Disagreement: {mean_disagreement:.4f}, Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        print("\nBottom 10 combinations based on mean pairwise disagreement:")
        for combination, mean_disagreement, acc, prec, rec, f1 in all_combinations_dis[-10:]:
            print(f"Models: {combination}, Mean Pairwise Disagreement: {mean_disagreement:.4f}, Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        # Print top 10 and bottom 10 combinations for double fault measure matrix
        print("\nTop 10 combinations based on mean double fault measure:")
        for combination, mean_double_fault, acc, prec, rec, f1 in all_combinations_df[:10]:
            print(f"Models: {combination}, Mean Double Fault Measure: {mean_double_fault:.4f}, Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        print("\nBottom 10 combinations based on mean double fault measure:")
        for combination, mean_double_fault, acc, prec, rec, f1 in all_combinations_df[-10:]:
            print(f"Models: {combination}, Mean Double Fault Measure: {mean_double_fault:.4f}, Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        print("\nAll disagreement:")
        for combination, mean_disagreement, acc, prec, rec, f1 in all_combinations_dis:
            print(f"Models: {combination}, Mean Pairwise Disagreement: {mean_disagreement:.4f}, Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        print("\nAll Double fault:")
        for combination, mean_double_fault, acc, prec, rec, f1 in all_combinations_df:
            print(f"Models: {combination}, Mean Double Fault Measure: {mean_double_fault:.4f}, Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

        #print("top diverse mean pairwise disagreement")
        #for result in results_dis:
        #    print(result)

        #print("top diverse mean df measure")
        #for result in results_df:
        #    print(result)
        #all_combinations = get_all_combinations_with_disagreement(normalized_disagreement_matrix)
        #for combination, mean_disagreement in all_combinations:
        #    print(f"Models: {combination}, Mean Pairwise Normalized Disagreement: {mean_disagreement:.4f}")
