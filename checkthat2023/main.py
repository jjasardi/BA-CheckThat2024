
import json
from pathlib import Path

import torch

from checkthat2023.tasks.task1a import load
from checkthat2023.kernel_stuff.token_alignment import TokenAlignmentDistance


def main(config):
    data_path = Path(config['data'])
    output_path = Path(config['output'])

    if not output_path.exists():
        raise ValueError(f"path \"{output_path}\" does not exist")

    task1a = load(data_folder=data_path, dev=config['dev'])

    alignment_kernel = TokenAlignmentDistance()

    d_train = alignment_kernel.distances(
        x=[
            s.tweet_text
            for s in task1a.train
        ],
    )
    torch.save(d_train, output_path / "train_dists.torch")

    d_test = alignment_kernel.distances(
        x=[
            s.tweet_text
            for s in task1a.dev_test
        ],
        y=[
            s.tweet_text
            for s in task1a.train
        ],
    )
    torch.save(d_test, output_path / "test_dists.torch")

    d_dev = alignment_kernel.distances(
        x=[
            s.tweet_text
            for s in task1a.dev
        ],
        y=[
            s.tweet_text
            for s in task1a.train
        ],
    )
    torch.save(d_dev, output_path / "dev_dists.torch")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f",
        dest="config_file",
        required=False,
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--config", "-c",
        dest="config_str",
        required=False,
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.config_file is not None:
        with args.config_file.open('r') as fin:
            config = json.load(fin)
    elif args.config_str is not None:
        config = json.loads(args.config_str)
    else:
        config = None

    main(config=config)

