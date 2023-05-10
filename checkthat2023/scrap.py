
from pathlib import Path

from checkthat2023.preprocessing.text import cardiffnlp_preprocess
from checkthat2023.tasks.task1a import load

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

import torch


data_path = Path('data')

task1a = load(data_folder=data_path, dev=True)


embed = TransformerWordEmbeddings("cardiffnlp/twitter-roberta-base")

s1 = Sentence(cardiffnlp_preprocess(task1a.train[0].tweet_text))
s2 = Sentence(cardiffnlp_preprocess(task1a.test[1].tweet_text))

embed.embed(s1)
embed.embed(s2)

e1 = torch.stack([t.embedding for t in s1.tokens])
e2 = torch.stack([t.embedding for t in s2.tokens])
