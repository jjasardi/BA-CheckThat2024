
from typing import List, Optional
from collections import Counter

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F

import ot

from tqdm import tqdm


class TokenAlignmentDistance:

    def __init__(
        self,
    ):
        self.model_name = "cardiffnlp/twitter-roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.idf = torch.ones(self.tokenizer.vocab_size)

    @staticmethod
    def __preprocess(text: str):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def fit_idf(self, corpus: List[str]):
        token_counts = Counter(
            token
            for sent in corpus
            for token in {
                t.item()
                for t in self.tokenizer(
                    self.__preprocess(sent),
                    return_tensors='pt',
                )['input_ids'][0]
            }
        )
        count_vec = torch.zeros(self.tokenizer.vocab_size)
        for i in range(self.tokenizer.vocab_size):
            count_vec[i] = token_counts[i]

        # add 1 for smoothing
        n_docs = len(corpus) + 1
        count_vec += 1

        self.idf = torch.log(n_docs / count_vec) + 1

    def token_embeds(self, txt: str):
        tokens = self.tokenizer(
            self.__preprocess(txt),
            return_tensors='pt',
        )
        with torch.no_grad():
            out = self.model(**tokens)

        token_weights = self.idf[tokens['input_ids']][0]
        token_weights /= token_weights.sum()
        token_embeds = F.normalize(out.last_hidden_state[0], dim=-1, p=2)

        return token_weights, token_embeds

    @staticmethod
    def __alignment_dist(w1, e1, w2, e2, decrease_precision: bool = False):
        sims = e1 @ e2.T
        cost = 1. - sims
        if decrease_precision:
            ot_emd = ot.emd(w1 / 100, w2 / 100, cost)
        else:
            ot_emd = ot.emd(w1, w2, cost)
        return (ot_emd * cost).sum()

    def distances(
        self,
        x: List[str],
        y: Optional[List[str]] = None,
        decrease_precision: bool = False,
    ):

        x_weights = []
        x_embeds = []
        for s in tqdm(x):
            w, e = self.token_embeds(s)
            x_weights.append(w)
            x_embeds.append(e)

        if y is None:
            dist_mat = torch.zeros((len(x), len(x)))
            for ix in tqdm(range(len(x))):
                for jx in range(ix, len(x)):
                    dist = self.__alignment_dist(
                        x_weights[ix],
                        x_embeds[ix],
                        x_weights[jx],
                        x_embeds[jx],
                        decrease_precision=decrease_precision,
                    )
                    dist_mat[ix, jx] = dist
                    dist_mat[jx, ix] = dist
        else:
            y_weights = []
            y_embeds = []
            for s in tqdm(y):
                w, e = self.token_embeds(s)
                y_weights.append(w)
                y_embeds.append(e)
            dist_mat = torch.zeros((len(x), len(y)))
            for ix in tqdm(range(len(x))):
                for jx in range(len(y)):
                    dist_mat[ix, jx] = self.__alignment_dist(
                        x_weights[ix],
                        x_embeds[ix],
                        y_weights[jx],
                        y_embeds[jx],
                        decrease_precision=decrease_precision,
                    )

        return dist_mat
