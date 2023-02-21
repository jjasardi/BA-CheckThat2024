
from pathlib import Path


def data_folder() -> Path:
    return Path(__file__).parents[1] / "data"


def task1a_folder() -> Path:
    return data_folder() / "task1A"


def task1b_folder() -> Path:
    return data_folder() / "task1B"
