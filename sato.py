import os
import json
import sys
from pathlib import Path
from allennlp.commands import main

sys.path += ['./scripts', './table_embedder/dataset_readers', './table_embedder/models']
from scripts.util import Util


def cmd_builder(setting, template_path, overrides):
    setting.pop('cuda_devices')
    for name, val in setting.items():
        os.environ[name] = val
    os.environ["ALLENNLP_DEBUG"] = "TRUE"

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        template_path,
        "-s", setting['out_dir'],
        "--include-package", "table_embedder",
        "-o", overrides,
    ]


if __name__ == "__main__":
    params = Util.load_yaml('exp/sato/sato.yml')
    if os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        params['cuda_devices'] = [int(elem) for elem in os.getenv('CUDA_VISIBLE_DEVICES').split(',')]
    for k, v in params['common'].items():
        params['train'][k] = v
    overrides = json.dumps({"distributed": {"cuda_devices": params['train']['cuda_devices']}}) if len(params['train']['cuda_devices'])>1 else {}
    cmd_builder(params['train'], "exp/sato/sato.jsonnet", overrides)
    main()


