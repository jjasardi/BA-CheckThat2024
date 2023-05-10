
from pathlib import Path

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

import torch
import torch.nn.functional as F

import numpy as np

import ot

from tqdm import tqdm

from checkthat2023.tasks.task1a import Task1A
from checkthat2023.preprocessing.text import (
    TweetNormalizer,
    cardiffnlp_preprocess,
)


def cost(e1, e2):
    with torch.no_grad():
        sims = e1 @ e2.T
        C = (1. - sims).cpu().numpy()

    ot_emd = ot.emd([], [], C, numThreads="max")
    return (ot_emd * C).sum()


def token_alignment_untrained(
    dataset: Task1A,
    base_model: str,
    output: Path,
):

    if "cardiffnlp" in base_model:
        prep_fn = cardiffnlp_preprocess
    else:
        prep_fn = TweetNormalizer()

    embed_model = TransformerWordEmbeddings(base_model)

    samples = {
        "train": dataset.train,
        "dev": dataset.dev,
        "test": dataset.test,
    }

    embeddings = {
        k: []
        for k in samples.keys()
    }

    print("RUN MODEL TO GET EMBEDDINGS")
    for split in ['train', 'dev', 'test']:
        for sample in tqdm(samples[split]):
            with torch.no_grad():
                s = Sentence(prep_fn(sample.tweet_text))
                embed_model.embed(s)
                es = torch.stack([t.embedding for t in s.tokens])
                es = F.normalize(es, dim=-1, p=2)
                embeddings[split].append(es)

    print("TRAIN KERNEL")
    k_train = np.zeros((len(dataset.train), len(dataset.train)))
    for ix in tqdm(range(len(dataset.train))):
        for jx in range(ix, len(dataset.train)):
            d = cost(embeddings['train'][ix], embeddings['train'][jx])
            k_train[ix, jx] = d
            k_train[jx, ix] = d

    print("DEV KERNEL")
    k_dev = np.zeros((len(dataset.dev), len(dataset.train)))
    for ix in tqdm(range(len(dataset.dev))):
        for jx in range(len(dataset.train)):
            k_dev[ix, jx] = cost(embeddings['dev'][ix], embeddings['train'][jx])

    print("TEST KERNEL")
    k_test = np.zeros((len(dataset.test), len(dataset.train)))
    for ix in tqdm(range(len(dataset.test))):
        for jx in range(len(dataset.train)):
            k_test[ix, jx] = cost(embeddings['test'][ix], embeddings['train'][jx])

    torch.save(obj=torch.tensor(k_train), f=output / "train_untrained_token_dist.torch")
    torch.save(obj=torch.tensor(k_dev), f=output / "dev_untrained_token_dist.torch")
    torch.save(obj=torch.tensor(k_test), f=output / "test_untrained_token_dist.torch")
