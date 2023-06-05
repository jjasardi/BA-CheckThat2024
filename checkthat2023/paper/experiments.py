
from pathlib import Path
import json

import numpy as np
from sklearn.svm import SVC

from checkthat2023.tasks.task1a import load
from checkthat2023.kernel_stuff.kernel import KernelList, KernelData
from checkthat2023.paper.plots import (
    ClfCollection,
    ClfInfo,
    roc_plots,
    pr_curves,
    optimize_threshold,
    print_fscores,
)


def kernel_classifiers(
    y_train: np.array,
    y_dev: np.array,
):
    base_kernels = {
        'img-untrained-kernel': KernelData.load_from(Path("kernel_data/img-untrained")),
        'text-ngram': KernelData.load_from(Path("kernel_data/ngram/")),
        'gpt-ngram': KernelData.load_from(Path("kernel_data/gpt-ngram")),
        'electra-kernel': KernelData.load_from(Path("kernel_data/electra")),
        'multi-modal-kernel': KernelData.load_from(Path("kernel_data/multimodal")),
    }

    kernels = {
        "submission": KernelList(name="submission", kernels=[
            base_kernels[s]
            for s in ["text-ngram", "gpt-ngram", "electra-kernel", "multi-modal-kernel"]
        ]).kernel_data(),
        "all-kernels": KernelList(
            name="all-kernel", kernels=list(base_kernels.values())).kernel_data(),
        **base_kernels,
    }

    systems = [
        "img-untrained-kernel",
        "text-ngram",
        "gpt-ngram",
        "electra-kernel",
        "multi-modal-kernel",
        "submission",
        "all-kernels",
    ]
    clf_infos = {}

    for s in systems:
        k = kernels[s]
        svm = SVC(
            C=1., kernel='precomputed', class_weight='balanced', random_state=0xdeadbeef)
        svm.fit(k.train, y_train)

        test_scores = svm.decision_function(k.test)
        base_preds = svm.predict(k.test)

        dev_scores = svm.decision_function(k.dev)
        th = optimize_threshold(y_dev, dev_scores)

        optimized_preds = test_scores >= th

        info = ClfInfo(
            name=s,
            test_scores=test_scores,
            test_preds_base=base_preds,
            test_preds_tuned=optimized_preds,
        )

        clf_infos[s] = info

    return ClfCollection(
        systems=systems,
        infos=clf_infos,
    )


def direct_classifiers():
    systems = [
        "gpt-answer",
        "electra-clf",
        "multi-modal-clf",
    ]
    clf_infos = {}

    electra_scores = np.load("preds/text_model_test_logits.npy")
    clf_infos['electra-clf'] = ClfInfo(
        name="electra-clf",
        test_scores=electra_scores[:, 1],
        test_preds_base=electra_scores.argmax(axis=1),
        test_preds_tuned=None,
    )

    multi_scores = np.load("preds/multimodal_test_logits.npy")
    clf_infos['multi-modal-clf'] = ClfInfo(
        name="multi-modal-clf",
        test_scores=multi_scores[:, 1],
        test_preds_base=multi_scores.argmax(axis=1),
        test_preds_tuned=None,
    )

    with open("preds/gpt-predictions.json", 'r') as fin:
        answers = json.load(fin)
    answers = np.array([
        d['class_label'] == 'Yes'
        for d in answers
    ]).astype(int)

    clf_infos["gpt-answer"] = ClfInfo(
        name="gpt-answer",
        test_scores=None,
        test_preds_tuned=None,
        test_preds_base=answers,
    )

    return ClfCollection(
        systems=systems,
        infos=clf_infos,
    )


def main():
    task1a = load(Path("./data"), dev=False)

    y_train = np.array([s.class_label for s in task1a.train]).astype(int)
    y_dev = np.array([s.class_label for s in task1a.dev]).astype(int)
    y_test = np.array([s.class_label for s in task1a.test]).astype(int)

    kernel_collection = kernel_classifiers(y_train=y_train, y_dev=y_dev)
    clf_collection = direct_classifiers()

    # roc_plots(collection=kernel_collection, y_test=y_test)
    # pr_curves(collection=kernel_collection, y_test=y_test)

    # roc_plots(collection=clf_collection, y_test=y_test)
    # pr_curves(collection=clf_collection, y_test=y_test)

    all_systems = [
        "gpt-answer",
        "img-untrained-kernel",
        "text-ngram",
        "gpt-ngram",
        "electra-kernel",
        "multi-modal-kernel",
        "submission",
        "all-kernels",
        "electra-clf",
        "multi-modal-clf",
    ]
    all_infos = {
        **kernel_collection.infos,
        **clf_collection.infos
    }
    full_collection = ClfCollection(systems=all_systems, infos=all_infos)

    roc_plots(full_collection, y_test=y_test, save_path="paper/all_roc.png")
    pr_curves(full_collection, y_test=y_test, save_path="paper/all_pr.png")

    print_fscores(full_collection, y_test=y_test)


if __name__ == "__main__":
    main()

