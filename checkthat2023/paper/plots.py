
from typing import List, Dict

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from matplotlib import pyplot as plt

import numpy as np


def roc_plots(
    systems: List[str],
    test_scores: Dict[str, np.array],
    dev_scores: Dict[str, np.array],
):
    pass
