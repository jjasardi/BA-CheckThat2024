from eval import get_predictions
import numpy as np
import csv
import time

def creae_submission_file(ids, predictions, model_label):
    class_labels = ["No", "Yes"]
    predicted_labels = [class_labels[idx] for idx in np.argmax(predictions, axis=1)]

    rows = []
    for i, sentence_id in enumerate(ids):
        row = {
            "id": sentence_id,
            "class_label": predicted_labels[i],
            "run_id": model_label
        }
        rows.append(row)

    timestamp = int(time.time())
    output_file = f"./checkthat2024/submission-samples/task1_english_{timestamp}_{model_label}.tsv"

    # Write the data to the TSV file
    with open(output_file, "w", newline="") as tsv_file:
        fieldnames = ["id", "class_label", "run_id"]
        writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    from checkthat2024.task1a import load
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("-l", "--model", dest="model_name", type=str, required=True)
    parser.add_argument("-n", "--model-label", dest="model_label", required=True)
    args = parser.parse_args()

    dataset = load(data_folder=args.data_folder, gold=True)
    x_test_gold = [s.text for s in dataset.test_gold]
    sentence_ids = [s.id for s in dataset.test_gold]
    predictions = get_predictions(model_name=args.model_name, x=x_test_gold, y=None)
    creae_submission_file(ids=sentence_ids, predictions=predictions, model_label=args.model_label)
