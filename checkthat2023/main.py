
import json
from pathlib import Path

from checkthat2023.tasks.task1a import load
from checkthat2023.finetune_multi import finetune


def main(config):
    data_path = Path(config['data'])
    output_path = Path(config['output'])

    if not output_path.exists():
        raise ValueError(f"path \"{output_path}\" does not exist")

    task1a = load(data_folder=data_path, dev=config['dev'])

    finetune(
        dataset=task1a,
        txt_model=config['txt_model'],
        img_model=config['img_model'],
        output_dir=output_path / "hf_out",
        log_dir=output_path / "hf_log",
        dev_mode=config['dev'],
    )


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

