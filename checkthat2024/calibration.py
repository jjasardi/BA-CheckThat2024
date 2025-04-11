from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from abc import abstractmethod
import numpy as np

# Code from von Däniken Pius

class CalibrationMethod:

    @abstractmethod
    def fit(self, y_scores, y_true):
        raise NotImplementedError

    @abstractmethod
    def transform(self, y_scores):
        raise NotImplementedError

class PlattScaling(CalibrationMethod):

    def __init__(self):
        self.logreg = LogisticRegression(
            C=1.,
            fit_intercept=True,
            penalty=None,
            class_weight=None,
            random_state=0xdeadbeef,
        )

    def fit(self, y_scores, y_true):
        self.logreg.fit(y_scores[:, np.newaxis], y_true)
        return self

    def transform(self, y_scores):
        probas = self.logreg.predict_proba(y_scores[:, np.newaxis])
        return probas[:, 1]


class Isotonic(CalibrationMethod):

    def __init__(self):
        self.iso = IsotonicRegression(
            y_min=0.,
            y_max=1.,
            increasing=True,
            out_of_bounds='clip',
        )

    def fit(self, y_scores, y_true):
        self.iso.fit(y_scores[:, np.newaxis], y_true)
        return self

    def transform(self, y_scores):
        scores = self.iso.predict(y_scores[:, np.newaxis])
        return scores
    

if __name__ == "__main__":
    from checkthat2024.task1a import load
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_folder", type=Path, required=True)
    parser.add_argument("-l", "--model", dest="model_name", type=str, required=True)
    parser.add_argument("-n", "--model-label", dest="model_label", required=True)
    args = parser.parse_args()

    dataset = load(data_folder=args.data_folder, gold=True)
    predictions = get_predictions(model_name=args.model_name, x=x_test_gold, y=None)