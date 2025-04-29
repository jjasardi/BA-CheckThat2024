import wandb
import json
import math

api = wandb.Api()

for run in api.runs(path="ba24/ba24-check-worthiness-estimation"):
    for i, row in run.history().iterrows():
        if "test.test_f1" in row and not math.isnan(row["test.test_f1"]):
            f1 = row["test.test_f1"]
            acc = row["test.test_accuracy"]
            precision = row["test.test_precision"]
            recall  = row["test.test_recall"]
            
            # Remove from config if exists
            if "test.test_accuracy" in run.config:
                del run.config["test.test_accuracy"]
            if "test.test_f1" in run.config:
                del run.config["test.test_f1"]
            if "test.test_precision" in run.config:
                del run.config["test.test_precision"]
            if "test.test_recall" in run.config:
                del run.config["test.test_recall"]
            run.update()

            # Add to summary
            run.summary["test.test_accuracy"] = acc
            run.summary["test.test_f1"] = f1
            run.summary["test.test_precision"] = precision
            run.summary["test.test_recall"] = recall
            run.update()
        
#         break
    # meta = json.load(run.file("wandb-metadata.json").download(replace=True))
    # if meta["program"] == "-m checkthat2024.finetune_text":
    #     data_index = meta["args"].index("--data") + 1
    #     data_folder = meta["args"][data_index]
    #     run.config["data"] = data_folder.replace(".", "").replace("/", "")

    #     model_index = meta["args"].index("--base-model") + 1
    #     model = meta["args"][model_index]
    #     run.config["model"] = model.replace("/", "-")
    #     run.update()



