
from dataclasses import dataclass
from typing import List, Dict, Optional

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_fscore_support,
    precision_recall_curve,
)

from matplotlib import pyplot as plt

import numpy as np


@dataclass
class ClfInfo:
    name: str
    test_preds_base: np.array
    test_scores: Optional[np.array] = None
    test_preds_tuned: Optional[np.array] = None


@dataclass
class ClfCollection:
    systems: List[str]
    infos: Dict[str, ClfInfo]


@dataclass
class RocPoint:
    tpr: float
    fpr: float


@dataclass
class PRPoint:
    precision: float
    recall: float
    f1: float


def roc_point(y_true: np.array, y_pred: np.array):
    true = y_true.astype(int)
    pred = y_pred.astype(int)

    tpr = (pred[true == 1] == 1).mean()
    fpr = (pred[true == 0] == 1).mean()
    return RocPoint(
        tpr=tpr,
        fpr=fpr,
    )


def pr_point(y_true: np.array, y_pred: np.array):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=1,
        average='binary',
    )
    return PRPoint(
        precision=p,
        recall=r,
        f1=f1,
    )


def optimize_threshold(y_true, y_scores):
    ps, rs, ths = precision_recall_curve(
        y_true=y_true,
        probas_pred=y_scores,
        pos_label=1,
    )

    f1s = (2 * ps * rs) / (ps + rs)
    f1s = f1s[:-1]

    not_nan = np.isfinite(f1s)

    return ths[not_nan][f1s[not_nan].argmax()]


def roc_plots(
    collection: ClfCollection,
    y_test: np.array,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots()
    ax.set_xlim(.0, 1.01)
    ax.set_ylim(.0, 1.01)
    fig.set_size_inches(10, 10)
    fig.set_tight_layout(True)

    ax.plot([0., 1.], [0., 1.], linestyle='--', label="random", c='b')

    base_rocs = []
    tuned_rocs = []

    # plot curves and points
    for s in collection.systems:
        info = collection.infos[s]
        if info.test_scores is not None:
            _ = RocCurveDisplay.from_predictions(
                y_true=y_test,
                y_pred=info.test_scores,
                name=s,
                ax=ax,
            )
            base_rocs.append(
                roc_point(y_true=y_test, y_pred=info.test_preds_base))
            if info.test_preds_tuned is not None:
                tuned_rocs.append(
                    roc_point(y_true=y_test, y_pred=info.test_preds_tuned))
        else:
            roc = roc_point(y_true=y_test, y_pred=info.test_preds_base)
            ax.scatter(
                x=[roc.fpr],
                y=[roc.tpr],
                marker='D',
                label=s,
                s=52,
            )

    # plot default operating points
    if len(base_rocs) > 0:
        ax.scatter(
            x=[r.fpr for r in base_rocs],
            y=[r.tpr for r in base_rocs],
            marker='x',
            c='k',
            label="Base Operating Point",
            s=52,
        )

    # plot tuned operating points
    if len(tuned_rocs) > 0:
        ax.scatter(
            x=[r.fpr for r in tuned_rocs],
            y=[r.tpr for r in tuned_rocs],
            marker='x',
            c='r',
            label="F1 Tuned Operating Point",
            s=52,
        )

    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def pr_curves(
    collection: ClfCollection,
    y_test: np.array,
    save_path: Optional[str] = None,
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
            marker='x',
            c='k',
            label="Base Operating Point",
            s=52,
        )

    # plot tuned operating points
    if len(tuned_pr) > 0:
        ax.scatter(
            x=[p.recall for p in tuned_pr],
            y=[p.precision for p in tuned_pr],
            marker='x',
            c='r',
            label="F1 Tuned Operating Point",
            s=52,
        )

    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def print_fscores(
    collection: ClfCollection,
    y_test: np.array,
):
    for s in collection.systems:
        info = collection.infos[s]

        default = pr_point(
            y_true=y_test,
            y_pred=info.test_preds_base,
        )
        if info.test_preds_tuned is not None:
            tuned = pr_point(
                y_true=y_test,
                y_pred=info.test_preds_tuned,
            )
        else:
            tuned = None
        if info.test_scores is not None:
            th = optimize_threshold(
                y_true=y_test,
                y_scores=info.test_scores,
            )
            opt_preds = (info.test_scores >= th).astype(int)
            opt = pr_point(
                y_true=y_test,
                y_pred=opt_preds,
            )
        else:
            opt = None

        line = f"{s} & {default.precision:.3f} & {default.recall:.3f} & {default.f1:.3f}"
        if tuned is not None:
            line += f" & {tuned.precision:.3f} & {tuned.recall:.3f} & {tuned.f1:.3f}"
        else:
            line += f" & - & - & -"
        if opt is not None:
            line += f" & {opt.precision:.3f} & {opt.recall:.3f} & {opt.f1:.3f}"
        else:
            line += f" & - & - & -"
        line += " \\\\"

        print(line)
