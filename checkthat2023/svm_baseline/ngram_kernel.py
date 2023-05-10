
from pathlib import Path

import torch

from sklearn.feature_extraction.text import TfidfVectorizer

from checkthat2023.preprocessing.text import TweetNormalizer
from checkthat2023.tasks.task1a import load


data_folder = Path('data')
kernel_folder = Path('kernel_data/ngram/')

task1a = load(data_folder, dev=False)

vectorizer = TfidfVectorizer(
    preprocessor=TweetNormalizer(),
    ngram_range=(1, 2),
    min_df=3,
    binary=True,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
)

x_train = [
    s.tweet_text
    for s in task1a.train
]
x_dev = [
    s.tweet_text
    for s in task1a.dev
]
x_test = [
    s.tweet_text
    for s in task1a.test
]

v_train = vectorizer.fit_transform(x_train)
v_dev = vectorizer.transform(x_dev)
v_test = vectorizer.transform(x_test)

torch.save(
    obj=torch.tensor((v_train @ v_train.T).todense()),
    f=kernel_folder / "train_ngram_sim.torch"
)
torch.save(
    obj=torch.tensor((v_dev @ v_train.T).todense()),
    f=kernel_folder / "dev_ngram_sim.torch"
)
torch.save(
    obj=torch.tensor((v_test @ v_train.T).todense()),
    f=kernel_folder / "test_ngram_sim.torch"
)
