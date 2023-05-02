
import torch
import torch.nn.functional as F

import ot

from flair.embeddings import TransformerEmbeddings
from flair.data import Sentence


class Embedder:

    def __init__(
        self,
        embedding_model: TransformerEmbeddings,
    ):
        self.embedding_model = embedding_model

    def __call__(self, text: str) -> torch.Tensor:
        s = Sentence(text)
        self.embedding_model.embed(s)
        embeds = torch.stack([t.embedding for t in s])
        return F.normalize(embeds, dim=1, p=2).to('cpu')


class TransportDist:

    def __call__(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:

        similarities = e1 @ e2.T
        cost = 1. - similarities

        # These are dummy probability distributions
        a = torch.zeros(e1.shape[0]) / e1.shape[0]
        b = torch.zeros(e2.shape[0]) / e2.shape[1]

        # optimal transport alignment plan
        ot_emd = ot.emd(a, b, cost)

        dist = (ot_emd * cost).sum()

        return dist
