# CheckThat2023

Developed using `python 3.10`.
Dependencies managed by [poetry](https://python-poetry.org/).

## Installation

* setup [poetry](https://python-poetry.org/)
* make sure you have `python 3.10` installed
* run `poetry install`

Alternatively all dependencies and their versions are listed in `pyproject.toml`.

## Fine-tune the text-only classifier

Assuming you have the **CheckThat2023 Task 1** data in the project root directory.
Run all commands from project root directory.

* `python -m checkthat2023.finetune_text --data ./data/ --base-model google/electra-base-discriminator --output ./model_dump --dev`

When the `--dev` flag is set, we will only load a small subset of the data and train for 1 epoch for testing. For full
training omit it.

When using `poetry`, you will need to run the above command with `poetry run` or in a `poetry shell`.