
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import classification_report

from checkthat2023.tasks.task1a import load
from checkthat2023.kernel_stuff.kernel import KernelList, KernelData

task1a = load(Path("./data"), dev=False)

y_train = [s.class_label for s in task1a.train]
y_dev = [s.class_label for s in task1a.dev]
y_test = [s.class_label for s in task1a.test]

kernels = {
    folder.name: KernelData.load_from(folder)
    for folder in Path("./kernel_data/").glob("*")
}

submission_kernel = KernelList(name="submission", kernels=[
    kernels[name]
    for name in ["electra", "gpt-ngram", "multimodal", "ngram"]
])

svm = SVC(C=1., kernel='precomputed', class_weight="balanced", random_state=0xdeadbeef)
svm.fit(submission_kernel.train(), y_train)

print(classification_report(y_true=y_test, y_pred=svm.predict(submission_kernel.test()), digits=3))
