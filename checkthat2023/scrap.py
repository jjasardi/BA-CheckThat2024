
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from checkthat2023.preprocessing.text import cardiffnlp_preprocess
from checkthat2023.tasks.task1a import load


matplotlib.use('QtAgg')

data_path = Path('data')
gpt_cache = Path('cache/gpt-cache.json')
electra_kernel = Path('kernel_data/electra')

task1a = load(data_folder=data_path, dev=False)

with gpt_cache.open('r') as fin:
    gpt_data = json.load(fin)

from checkthat2023.img_kernel_untrained import Embedder

e = Embedder()

es = e.embeddings(task1a.train[:2])
