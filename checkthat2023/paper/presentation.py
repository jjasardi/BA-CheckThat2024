
from typing import Optional, Set
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context

from sklearn.metrics import PrecisionRecallDisplay

from checkthat2023.tasks.task1a import load
from checkthat2023.paper.experiments import (
    kernel_classifiers,
    direct_classifiers,
)
from checkthat2023.paper.plots import ClfCollection, pr_point


def pr_curves(
        collection: ClfCollection,
        y_test: np.array,
        save_path: Optional[str] = None,
        highlight: Optional[Set[str]] = None,
):
    fig, ax = plt.subplots()
    ax.set_xlim(.0, 1.01)
    ax.set_ylim(.0, 1.01)
    fig.set_size_inches(10, 10)
    fig.set_tight_layout(True)

    # plot lines of constant F1
    for fixed_f1 in np.linspace(.1, .9, num=9):
        ps = np.linspace(.01, .99, num=98)
        rs = (- fixed_f1 * ps) / (fixed_f1 - 2 * ps)
        ax.plot(rs[rs > 0], ps[rs > 0], c='k', alpha=.3)
        txt_x = .95
        txt_y = (- fixed_f1 * 0.95) / (fixed_f1 - 2 * 0.95) + .01
        ax.text(x=txt_x, y=txt_y, s=f"F1 {fixed_f1:.1f}")

    base_pr = []
    tuned_pr = []

    # plot curves and points
    for s in collection.systems:
        info = collection.infos[s]
        if info.test_scores is not None:
            _ = PrecisionRecallDisplay.from_predictions(
                y_true=y_test,
                y_pred=info.test_scores,
                name=s,
                ax=ax,
                alpha=.4 if (highlight is not None and s not in highlight) else 1.,
            )
            base_pr.append(
                pr_point(y_true=y_test, y_pred=info.test_preds_base))
            if info.test_preds_tuned is not None:
                tuned_pr.append(
                    pr_point(y_true=y_test, y_pred=info.test_preds_tuned))
        else:
            pr = pr_point(y_true=y_test, y_pred=info.test_preds_base)
            ax.scatter(
                x=[pr.recall],
                y=[pr.precision],
                marker='D',
                label=s,
                s=52,
            )

    # plot default operating points
    if len(base_pr) > 0:
        ax.scatter(
            x=[p.recall for p in base_pr],
            y=[p.precision for p in base_pr],
            marker='o',
            c='k',
            label="Base Operating Point",
            s=70,
        )

    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    task1a = load(Path("./data"), dev=False)

    y_train = np.array([s.class_label for s in task1a.train]).astype(int)
    y_dev = np.array([s.class_label for s in task1a.dev]).astype(int)
    y_test = np.array([s.class_label for s in task1a.test]).astype(int)

    kernel_collection = kernel_classifiers(y_train=y_train, y_dev=y_dev)
    clf_collection = direct_classifiers()

    all_systems = [
        "gpt-answer",
        "text-ngram",
        "gpt-ngram",
        # "electra-kernel",
        # "multi-modal-kernel",
        "submission",
        "electra-clf",
        "multi-modal-clf",
    ]
    all_infos = {
        **kernel_collection.infos,
        **clf_collection.infos
    }
    full_collection = ClfCollection(systems=all_systems, infos=all_infos)

    pr_curves(
        full_collection,
        y_test=y_test,
        save_path="presentation/all.png",
        highlight=None,
    )
    pr_curves(
        full_collection,
        y_test=y_test,
        save_path="presentation/ngrams.png",
        highlight={"text-ngram", "gpt-ngram"},
    )
    pr_curves(
        full_collection,
        y_test=y_test,
        save_path="presentation/dl.png",
        highlight={"submission", "electra-clf", "multi-modal-clf"},
    )
    pr_curves(
        full_collection,
        y_test=y_test,
        save_path="presentation/multi-modal-only.png",
        highlight={"multi-modal-clf"},
    )


if __name__ == "__main__":
    main()

