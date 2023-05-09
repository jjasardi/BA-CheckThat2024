
from dataclasses import dataclass
from pathlib import Path

import torch

import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize

from sklearn.svm import SVC
from sklearn.metrics import hinge_loss

from checkthat2023.tasks.task1a import load
from checkthat2023.evaluation import build_prediction_samples, evaluate

data_path = Path('./data')

task1a = load(data_folder=data_path, dev=False)

y_train = [s.class_label for s in task1a.train]
y_dev = [s.class_label for s in task1a.dev]
y_test = [s.class_label for s in task1a.dev_test]

K_ngram_train = torch.load('output/train_ngram.torch')
K_ngram_test = torch.load('output/test_ngram.torch')
K_ngram_dev = torch.load('output/dev_ngram.torch')

K_img_train = torch.load('output/train_img_sim.torch')
K_img_test = torch.load('output/test_img_sim.torch')
K_img_dev = torch.load('output/dev_img_sim.torch')

D_tok_train = torch.load('output/train_dists.torch')
D_tok_test = torch.load('output/test_dists.torch')
D_tok_dev = torch.load('output/dev_dists.torch')

D_tok_idf_train = torch.load('output/train_idf_dists.torch')
D_tok_idf_test = torch.load('output/test_idf_dists.torch')
D_tok_idf_dev = torch.load('output/dev_idf_dists.torch')


def gaussian_kernel(dists, gamma=1.):
    return torch.exp(-gamma*dists)


class OptimizationHelper:

    @dataclass
    class Args:
        svm_c: float
        gamma_tok: float
        gamma_tok_idf: float
        logit_ngram: float
        logit_img: float
        logit_tok: float
        logit_tok_idf: float
        p_ngram: float
        p_img: float
        p_tok: float
        p_tok_idf: float

        @staticmethod
        def initial() -> 'OptimizationHelper.Args':
            return OptimizationHelper.Args(
                svm_c=1.,
                gamma_tok=1.,
                gamma_tok_idf=1.,
                logit_ngram=0.,
                logit_img=0.,
                logit_tok=0.,
                logit_tok_idf=0.,
                p_ngram=.25,
                p_img=.25,
                p_tok=.25,
                p_tok_idf=.25,
            )

        def to_arr(self) -> np.array:
            res = np.zeros(7)
            res[0] = self.svm_c
            res[1] = self.gamma_tok
            res[2] = self.gamma_tok_idf
            res[3] = self.logit_ngram
            res[4] = self.logit_img
            res[5] = self.logit_tok
            res[6] = self.logit_tok_idf
            return res

        @staticmethod
        def from_arr(x) -> 'OptimizationHelper.Args':
            ps = softmax(x[3:])
            return OptimizationHelper.Args(
                svm_c=x[0],
                gamma_tok=x[1],
                gamma_tok_idf=x[2],
                logit_ngram=x[3],
                logit_img=x[4],
                logit_tok=x[5],
                logit_tok_idf=x[6],
                p_ngram=ps[0],
                p_img=ps[1],
                p_tok=ps[2],
                p_tok_idf=ps[3],
            )

    def __init__(self):
        pass

    @staticmethod
    def x0() -> np.array:
        return OptimizationHelper.Args.initial().to_arr()

    @staticmethod
    def build_kernel(args: 'OptimizationHelper.Args', split: str):
        if split == 'train':
            k = \
                args.p_ngram * K_ngram_train + \
                args.p_img * K_img_train + \
                args.p_tok * gaussian_kernel(D_tok_train, gamma=args.gamma_tok) + \
                args.p_tok_idf * gaussian_kernel(D_tok_idf_train, args.gamma_tok_idf)
        elif split == "test":
            k = \
                args.p_ngram * K_ngram_test + \
                args.p_img * K_img_test + \
                args.p_tok * gaussian_kernel(D_tok_test, gamma=args.gamma_tok) + \
                args.p_tok_idf * gaussian_kernel(D_tok_idf_test, args.gamma_tok_idf)
        elif split == "dev":
            k = \
                args.p_ngram * K_ngram_dev + \
                args.p_img * K_img_dev + \
                args.p_tok * gaussian_kernel(D_tok_dev, gamma=args.gamma_tok) + \
                args.p_tok_idf * gaussian_kernel(D_tok_idf_dev, args.gamma_tok_idf)
        else:
            raise ValueError(f"unknown split \"{split}\","
                             f" use one of ['train', 'test', 'dev'")

        return k

    @staticmethod
    def evaluate_solution(x: np.array):
        args = OptimizationHelper.Args.from_arr(x)
        k_train = OptimizationHelper.build_kernel(args, split="train")
        k_test = OptimizationHelper.build_kernel(args, split="test")

        svm = SVC(C=args.svm_c, kernel='precomputed', class_weight='balanced')
        svm.fit(k_train, y_train)

        y_pred = svm.predict(k_test)

        return evaluate(task1a.dev_test, build_prediction_samples(
            task1a.dev_test,
            y_pred,
        ))

    def __call__(self, x: np.array):

        args = OptimizationHelper.Args.from_arr(x)
        k_train = OptimizationHelper.build_kernel(args, split='train')
        k_dev = OptimizationHelper.build_kernel(args, split="dev")

        svm = SVC(C=args.svm_c, kernel='precomputed', class_weight='balanced')
        svm.fit(k_train, y_train)

        s_dev = svm.decision_function(k_dev)

        return hinge_loss(y_true=y_dev, pred_decision=s_dev)


opt_fn = OptimizationHelper()
opt_res = minimize(
    fun=opt_fn,
    x0=opt_fn.x0(),
    method='Nelder-Mead',
)

print(opt_fn.evaluate_solution(opt_res.x))
