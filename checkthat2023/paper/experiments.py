
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from matplotlib import pyplot as plt

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from checkthat2023.tasks.task1a import load
from checkthat2023.kernel_stuff.kernel import KernelList, KernelData

task1a = load(Path("./data"), dev=False)

y_train = [s.class_label for s in task1a.train]
y_dev = [s.class_label for s in task1a.dev]
y_test = [s.class_label for s in task1a.test]


base_kernels = {
    folder.name: KernelData.load_from(folder)
    for folder in Path("./kernel_data/").glob("*")
}

submission_kernel = KernelList(name="submission", kernels=[
    base_kernels[name]
    for name in ["electra", "gpt-ngram", "multimodal", "ngram"]
]).kernel_data()

kernels = {
    submission_kernel.name: submission_kernel,
    "all": KernelList(name="all", kernels=list(base_kernels.values())).kernel_data(),
    **base_kernels,
}


fig, ax = plt.subplots()
ax.set_xlim(.0, 1.01)
ax.set_ylim(.0, 1.01)
fig.set_size_inches(10, 10)
fig.set_tight_layout(True)

ax.plot([0., 1.], [0., 1.], linestyle='--', label="random", c='b')

for name, kernel in kernels.items():
    svm = SVC(C=1., kernel='precomputed', class_weight="balanced", random_state=0xdeadbeef)
    svm.fit(kernel.train, y_train)

    _ = RocCurveDisplay.from_predictions(
        y_true=y_test,
        y_pred=svm.decision_function(kernel.test),
        name=name,
        ax=ax,
    )


electra_logits = np.load("./preds/text_model_test_logits.npy")
_ = RocCurveDisplay.from_predictions(
    y_true=y_test,
    y_pred=electra_logits[:, 1],
    name="electra-logits",
    ax=ax,
)

multimodal_logits = np.load('./preds/multimodal_test_logits.npy')
_ = RocCurveDisplay.from_predictions(
    y_true=y_test,
    y_pred=multimodal_logits[:, 1],
    name="multi-modal-logits",
    ax=ax,
)

ax.legend()

plt.show()
