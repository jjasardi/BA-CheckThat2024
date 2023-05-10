
from typing import Union, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class KernelData:
    train: Union[np.ndarray, torch.tensor]
    dev: Union[np.ndarray, torch.tensor]
    test: Union[np.ndarray, torch.tensor]
    name: Optional[str] = None

    @staticmethod
    def load_from(folder: Path) -> 'KernelData':

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
                             f"when loading kernel from '{folder}'")

        return KernelData(**arg_dict)
