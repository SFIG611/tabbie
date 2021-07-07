import os
import json
import sys
import argparse
import tempfile
from pathlib import Path
from allennlp.commands import main
from scripts.util import Util

sys.path += ['./scripts', './table_embedder/dataset_readers', './table_embedder/models']
from scripts.util import Util
from scripts.pretrain import Pretrain


def cmd_builder(params, overrides):
    params.pop('cuda_devices')
    for name, val in params.items():
        print(name, val)
        os.environ[name] = val
    sys.argv = [
        "allennlp",  # command name, not used by main
        "predict",
        params['model_path'],
        params['pred_path'],
        "--output-file", params['out_pred_path'],
        "--include-package", "table_embedder",
        "--predictor", "cell_predictor",
        "--cuda-device", 0,
        "--batch-size", str(params['batch_size']),
        "-o", overrides,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--test_csv_dir", help="input csv dir for validation", default="./data/ft_cell/test_csv")
    parser.add_argument("--test_csv_dir", help="input csv dir for validation", default="./data/ft_col/test_csv")
    parser.add_argument("--model_path", help="model file path", default="./out_model/model.tar.gz")
    # parser.add_argument("--test_label_path", help="label for test dataset if available", default="./data/ft_cell/test_label.csv")
    parser.add_argument("--test_label_path", help="label for test dataset if available", default="./data/ft_col/test_label.csv")
    parser.add_argument("--out_pred_path", help="output json dataset path", default="./out_model/pred_test.jsonl")
    parser.add_argument("--out_pred_dir", help="output json dataset path", default="./out_model/")
    parser.add_argument("--config", help="config file for tabbie", default="./exp/ft_cell/cell_pred.yml")
    args = parser.parse_args()

    # get label type
    label_type = None
    if args.test_label_path is not None:
        label_type = Util.get_label_type(args.test_label_path)

    # dump jsonl test data
    tmpdir = tempfile.TemporaryDirectory()
    Util.csvdir_to_jsonl(Path(args.test_csv_dir), Path(tmpdir.name)/'test.jsonl', label_path=args.test_label_path, label_type=label_type)

    params = Util.load_yaml(args.config)
    params['model_path'] = args.model_path
    params['pred_path'] = str(Path(tmpdir.name)/'test.jsonl')
    params['out_pred_path'] = args.out_pred_path
    params['learn_type'] = 'pred'

    overrides = json.dumps({'trainer': {'opt_level': 'O0'}})
    cmd_builder(params, overrides)
    main()

    lines = Util.load_lines(params['out_pred_path'])
    Util.ft_jsonl_to_html(lines, Path(args.out_pred_dir) / '{}_html'.format(label_type))



