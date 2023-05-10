
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

import torch

from checkthat2023.tasks.task1a import load

data_path = Path("data")
gpt_cache = Path("cache/gpt-cache.json")

kernel_folder = Path('kernel_data/gpt-ngram')

with gpt_cache.open('r') as fin:
    gpt_data = json.load(fin)


task1a = load(data_folder=data_path, dev=False)

x_train = [
    gpt_data[s.id]['variables']['REASON']
    for s in task1a.train
]
x_dev = [
    gpt_data[s.id]['variables']['REASON']
    for s in task1a.dev
]
x_test = [
    gpt_data[s.id]['variables']['REASON']
    for s in task1a.test
]

gpt_preds = [
    {
        "tweet_id": s.id,
        "class_label": gpt_data[s.id]['variables']['ANSWER'].strip(),
        "run_id": "gpt"
    }
    for s in task1a.test
]

with open('./preds/gpt-predictions.json', 'w') as fout:
    json.dump(obj=gpt_preds, fp=fout, indent=2)


gpt_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=3,
    binary=True,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
)

v_train = gpt_vectorizer.fit_transform(x_train)
v_dev = gpt_vectorizer.transform(x_dev)
v_test = gpt_vectorizer.transform(x_test)

torch.save(
    obj=torch.tensor((v_train @ v_train.T).todense()),
    f=kernel_folder / "train_gpt_ngram_sim.torch"
)
torch.save(
    obj=torch.tensor((v_dev @ v_train.T).todense()),
    f=kernel_folder / "dev_gpt_ngram_sim.torch"
)
torch.save(
    obj=torch.tensor((v_test @ v_train.T).todense()),
    f=kernel_folder / "test_gpt_ngram_sim.torch"
)
