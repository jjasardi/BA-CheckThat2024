
from typing import Tuple, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from checkthat2023.preprocessing.text import TweetNormalizer


class TfIdfKernel:

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 3,
        binary: bool = True,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.binary = binary

        self.vectorizer = TfidfVectorizer(
            preprocessor=TweetNormalizer(),
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            binary=self.binary,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
        )

    def fit(self, corpus: List[str]):
        self.vectorizer.fit(corpus)
        return self

    def kernel(self, x: List[str], y: Optional[List[str]] = None):
        xv = self.vectorizer.transform(x)
        if y is None:
            yv = xv
        else:
            yv = self.vectorizer.transform(y)

        return np.array((xv @ yv.T).todense())
