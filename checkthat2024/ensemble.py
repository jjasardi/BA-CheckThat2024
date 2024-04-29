from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from checkthat2024.finetune_text import TorchDataset

from checkthat2024.task1a import Task1A, load

from pathlib import Path
import numpy as np
from scipy import stats

import wandb

def ensemble(
    dataset: Task1A,
    models_names: list[str],
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
    for i, model_name in enumerate(models_names):
        model = AutoModelForSequenceClassification.from_pretrained("./model_dump/CT_24/" + model_name + "/text_model")
        tokenizer = AutoTokenizer.from_pretrained("./model_dump/CT_24/" + model_name + "/text_model")
        
        test = TorchDataset.from_samples(x_test, y_test, tokenizer)
        trainer = Trainer(model=model)
        all_predictions.append(trainer.predict(test_dataset=test).predictions)
    
    
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
    



if __name__ == "__main__":
    from checkthat2024.task1a import load
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("-l", "--models", dest="models", nargs="+", type=str, required=True)
    parser.add_argument("-v", "--voting", dest="voting", type=str, required=False)

    args = parser.parse_args()
    if len(args.models) < 2:
        print("please provide more than one model")
        sys.exit()

    model = f"ensemble-{'-'.join(args.models)}"
    config = {"data": args.data_folder, "model": model, "voting": args.voting}

    wandb.init(
        name="ensemble",
        group=f"ensemble-{args.data_folder}",
        config=config,
    )
    ensemble(
        dataset=load(data_folder=args.data_folder),
        models_names=args.models,
        voting=args.voting,
    )
