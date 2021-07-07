import os
import sys
import json
from pathlib import Path
from allennlp.commands import main
sys.path += ['./scripts', './table_embedder/dataset_readers', './table_embedder/models']
from scripts.util import Util
from scripts.pretrain import Pretrain


def cmd_builder(params, overrides):
    sys.argv = [
        "allennlp",  # command name, not used by main
        "predict",
        params['model_path'],
        params['pred_path'],
        "--output-file", str(Path(params['out_dir'])/params['out_pred_name']),
        "--include-package", "table_embedder",
        "--predictor", "predictor",
        "--cuda-device", 0,
        "--batch-size", str(params['batch_size']),
        "-o", overrides,
    ]


def setup_env_variables(params):
    os.environ["ALLENNLP_DEBUG"] = "TRUE"
    for name, val in params.items():
        print(name, val)
        if isinstance(name, list):
            continue
        os.environ[name] = val


if __name__ == "__main__":
    # initialize
    params = Util.load_yaml('./exp/pretrain/26600K_mix.yml')  # './exp/pretrain/26600K_freq.yml'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, params['cuda_devices']))
    del params['cuda_devices']
    setup_env_variables(params)

    # predict
    overrides = json.dumps({'dataset_reader': {'type': 'preprocess'}, 'trainer': {'opt_level': 'O0'}})
    cmd_builder(params, overrides)
    main()

    # dump pred
    Pretrain.dump_pred(Path(params['out_dir'])/params['out_pred_name'], Path(params['out_dir']))



