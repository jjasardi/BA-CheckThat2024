
from pathlib import Path

from transformers import (
    ElectraForSequenceClassification,
    ElectraTokenizer,
)
from transformers.activations import get_activation

import torch
import torch.nn as nn
import torch.nn.functional as F

from checkthat2023.tasks.task1a import Task1A

from tqdm import tqdm


class ElectraEmbeddingModel(nn.Module):

    def __init__(self, electra_clf: ElectraForSequenceClassification):
        super().__init__()
        self.electra_clf = electra_clf

    def forward(self, **kwargs):
        # from code ElectraForSequenceClassification
        model_out = self.electra_clf.electra(**kwargs)
        features = model_out[0]

        # from code ElectraClassificationHead
        x = features[:, 0, :]
        x = self.electra_clf.classifier.dropout(x)
        x = self.electra_clf.classifier.dense(x)
        x = get_activation('gelu')(x)
        x = self.electra_clf.classifier.dropout(x)
        return x


def electra_sim(
    dataset: Task1A,
    output: Path
):
    tok = ElectraTokenizer.from_pretrained(output / "text_model")
    mod = ElectraForSequenceClassification.from_pretrained(output / "text_model")

    embedding_model = ElectraEmbeddingModel(mod).eval()

    samples = {
        "train": dataset.train,
        "dev": dataset.dev,
        "test": dataset.test,
    }

    embeddings = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for split in ["train", "dev", "test"]:
        with torch.no_grad():
            for s in tqdm(samples[split]):
                i = tok(s.tweet_text, truncation=True, padding=True, return_tensors='pt')
                embeddings[split].append(embedding_model(**i))

    with torch.no_grad():
        embeddings = {
            k: F.normalize(torch.cat(vs), dim=-1, p=2)
            for k, vs in embeddings.items()
        }

    with torch.no_grad():
        k_train = embeddings['train'] @ embeddings['train'].T
        k_dev = embeddings['dev'] @ embeddings['train'].T
        k_test = embeddings['test'] @ embeddings['train'].T

    torch.save(obj=k_train, f=output / "train_electra_sim.torch")
    torch.save(obj=k_dev, f=output / "dev_electra_sim.torch")
    torch.save(obj=k_test, f=output / "test_electra_sim.torch")
