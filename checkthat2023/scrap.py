
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    f1_score,
)

import numpy as np

import torch

from checkthat2023.tasks.task1a import load
from checkthat2023.kernel_stuff.kernel import KernelData, KernelList


matplotlib.use('QtAgg')

data_path = Path('data')
kernel_folder = Path('kernel_data/')

task1a = load(data_folder=data_path, dev=False)

y_train = [
    s.class_label
    for s in task1a.train
]
y_dev = [
    s.class_label
    for s in task1a.dev
]

fig, ax = plt.subplots()

kernels = [
    KernelData.load_from(kf)
    for kf in kernel_folder.glob("*")
]

kl = KernelList(kernels=kernels, name="avg-kernel")

for c in np.logspace(-3, 3, base=10, num=7):
    svm = SVC(
        C=c,
        kernel='precomputed',
        class_weight='balanced',
        random_state=0xdeadbeef,
    )
    svm.fit(kl.train(), y_train)
    scores = svm.decision_function(kl.dev())
    ps, rs, ths = precision_recall_curve(
        y_true=y_dev,
        probas_pred=scores,
        pos_label=1,
    )
    f1 = (2. * ps * rs) / (ps + rs)
    ix = f1.argmax()
    base_f1 = f1_score(
        y_true=y_dev,
        y_pred=svm.predict(kl.dev()),
        pos_label=1,
        average='binary',
    )
    print(f"C: {c:.3f}\tbest_thresh: {ths[ix]:.3f}\tF1: {f1[ix]:.3f}\tF1-base: {base_f1:.3f}")
