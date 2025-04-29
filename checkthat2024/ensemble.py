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
from checkthat2024.eval import get_predictions

import wandb

def ensemble(
    dataset: Task1A,
    model_names: list[str],
    processed_models: list[int]=[],
    voting: str='hard',
):
    x_test = [
            s.text
            for s in dataset.test
        ]
    y_test = [
        s.class_label if hasattr(s, 'class_label') else None
        for s in dataset.test
    ]

    all_predictions = []
    for i, model_name in enumerate(model_names):     
        test_texts = x_test
        if i in processed_models:
            test_texts = process_texts(x_test)

        model_predictions = get_predictions(model_name, test_texts, y_test)
        all_predictions.append(model_predictions)
    
    
    if voting =='hard':
        all_predicted_labels = [np.argmax(prediction, axis=1) for prediction in all_predictions]
        majority_vote = stats.mode(all_predicted_labels, keepdims=False)[0]
        
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
    from checkthat2024.eval import models_disagreement_matrix, visualize_matrix

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
        name=f"ensemble-{'-'.join(args.model_names)}",
        group=f"ensemble-{args.data_folder}",
        config=config,
    )
    ensemble(
        dataset=load(data_folder=args.data_folder),
        model_names=args.model_names,
        voting=args.voting,
        processed_models=args.processed_models,
    )

    if len(args.model_labels) == len(args.model_names):
        logits = []
        for model in args.model_names:
            logits.append(np.load("./model_dump/CT_24/" + model + "/text_model_test_logits.npy"))
        disagreement_matrix = models_disagreement_matrix(logits)
        visualize_matrix(disagreement_matrix, args.model_labels, "Disagreement of model predictions")
