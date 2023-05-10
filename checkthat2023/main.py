
import json
from pathlib import Path

from checkthat2023.tasks.task1a import load
from checkthat2023.finetune_multi import finetune as finetune_multi
from checkthat2023.finetune_text import finetune as finetune_text
from checkthat2023.electra_kernel import electra_sim


def main(config):
    data_path = Path(config['data'])
    output_path = Path(config['output'])
    mode = config['mode']
    dev_mode = config['dev']

    if not output_path.exists():
        raise ValueError(f"path \"{output_path}\" does not exist")

    task1a = load(data_folder=data_path, dev=dev_mode)

    if mode == "finetune_text":
        finetune_text(
            dataset=task1a,
            base_model=config['finetune_text']['base_model'],
            output_dir=output_path,
            dev_mode=dev_mode,
        )
    if mode == "electra_kernel":
        electra_sim(
            dataset=task1a,
            output=output_path,
        )
    else:
        raise ValueError(f"unknown experiment mode '{mode}'")


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

