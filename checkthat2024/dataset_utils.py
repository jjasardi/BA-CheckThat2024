import torch
from torch.utils.data import Dataset as TDataset
from typing import List, Optional
from transformers import AutoTokenizer

class TorchDataset(TDataset):

    def __init__(
        self,
        torch_data: dict
    ):
        self.torch_data = torch_data

    def __len__(self) -> int:
        return self.torch_data['input_ids'].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            k: v[idx]
            for k, v in self.torch_data.items()
        }

    @staticmethod
    def from_samples(
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer: AutoTokenizer,
    ) -> 'TorchDataset':
        torch_data = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt")
        if labels is not None:
            torch_data['labels'] = torch.LongTensor(labels)
        return TorchDataset(torch_data)