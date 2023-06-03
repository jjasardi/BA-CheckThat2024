
from typing import Union, Optional, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _load(folder: Path) -> dict:
    arg_dict = {
        "name": folder.name
    }

    for p in folder.glob('*.torch'):
        for split in ['train', 'test', 'dev']:
            if p.name.startswith(split):
                arg_dict[split] = torch.load(p)

    missing = {
        arg
        for arg in ['train', 'test', 'dev', 'name']
        if arg not in arg_dict.keys()
    }

    if len(missing) > 0:
        raise ValueError(f"missing: '{missing}' "
                         f"when loading kernel or dist data from '{folder}'")

    return arg_dict


@dataclass(frozen=True)
class KernelData:
    train: Union[np.ndarray, torch.tensor]
    dev: Union[np.ndarray, torch.tensor]
    test: Union[np.ndarray, torch.tensor]
    name: Optional[str] = None

    @staticmethod
    def load_from(folder: Path) -> 'KernelData':
        return KernelData(**_load(folder))


def rbf(dists, gamma: float):
    return torch.exp(-gamma*dists)


@dataclass(frozen=True)
class DistData:
    train: Union[np.ndarray, torch.tensor]
    dev: Union[np.ndarray, torch.tensor]
    test: Union[np.ndarray, torch.tensor]
    name: Optional[str] = None

    @staticmethod
    def load_from(folder: Path) -> 'DistData':
        return DistData(**_load(folder))

    def rbf(self, gamma: float) -> 'ḰernelData':
        return KernelData(
            train=rbf(self.train, gamma),
            dev=rbf(self.dev, gamma),
            test=rbf(self.test, gamma),
            name=self.name,
        )


class KernelList:

    def __init__(self, kernels: List[KernelData], name: str):
        self.name = name
        self.kernels = kernels
        self.train_ = torch.stack([k.train for k in self.kernels], dim=-1)
        self.dev_ = torch.stack([k.dev for k in self.kernels], dim=-1)
        self.test_ = torch.stack([k.test for k in self.kernels], dim=-1)

    def __weighted_avg(self, kern_mat, w: Optional[torch.tensor] = None):
        if w is None:
            w = torch.ones(len(self.kernels), dtype=self.train_.dtype) / len(self.kernels)
        return kern_mat @ w

    def train(self, w: Optional[torch.tensor] = None):
        return self.__weighted_avg(self.train_, w)

    def dev(self, w: Optional[torch.tensor] = None):
        return self.__weighted_avg(self.dev_, w)

    def test(self, w: Optional[torch.tensor] = None):
        return self.__weighted_avg(self.test_, w)

    def kernel_data(self, w: Optional[torch.tensor] = None):
        return KernelData(
            name=self.name,
            train=self.train(w),
            dev=self.dev(w),
            test=self.test(w),
        )
