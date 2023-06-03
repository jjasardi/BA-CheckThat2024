
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
    hinge_loss,
    log_loss,
)
from sklearn.model_selection import ParameterSampler
from sklearn.utils.fixes import loguniform

import numpy as np

import torch

from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import softmax

from checkthat2023.tasks.task1a import load
from checkthat2023.kernel_stuff.kernel import KernelData, KernelList, DistData


matplotlib.use('QtAgg')

data_path = Path('data')
kernel_folder = Path('kernel_data/')
dist_folder = Path("dist_data/")

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
dist = DistData.load_from(dist_folder / "token-alignment")

kl = KernelList(kernels=kernels + [dist.rbf(gamma=10.)], name="avg-kernel")


def experiment(
    svm_c: float,
    gamma: float,
    kernel_weights: np.array,
):
    kernel_list = KernelList(kernels + [dist.rbf(gamma)], name="")
    svm = SVC(
        C=svm_c,
        kernel='precomputed',
        class_weight='balanced',
        random_state=0xdeadbeef,
        probability=True,
    )
    w = torch.tensor(kernel_weights)
    k_train = kernel_list.train(w)
    k_dev = kernel_list.dev(w)

    svm.fit(k_train, y_train)
    # f1 = f1_score(
    #     y_true=y_dev,
    #     y_pred=svm.predict(k_dev),
    #     pos_label=1,
    #     average='binary',
    # )
    # return -f1
    # return hinge_loss(y_true=y_dev, pred_decision=svm.decision_function(k_dev))
    return log_loss(y_true=y_dev, y_pred=svm.predict_proba(k_dev))


def wrapper(x: np.array):
    return experiment(
        svm_c=np.exp(x[0]),
        gamma=np.exp(x[1]),
        kernel_weights=softmax(x[2:]),
    )

x0 = np.zeros(len(kernels) + 1 + 2)

opt_res = minimize(
    fun=wrapper,
    x0=x0,
    method="Nelder-Mead",
    options={'verbose': 3}
)

