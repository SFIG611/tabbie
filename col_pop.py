import os
import json
import shutil
import sys
from allennlp.commands import main

sys.path += ['./_util', './downstream', './scripts', './table_embedder/dataset_readers', './table_embedder/models']
from scripts.util import Util


def cmd_builder(setting, config_file, overrides):
    os.environ["ALLENNLP_DEBUG"] = "TRUE"
    for name, val in setting.items():
        os.environ[name] = val

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", os.environ['out_dir'],
        "--include-package", "table_embedder",
        "-o", overrides,
    ]


if __name__ == "__main__":
    setting = Util.load_yaml('exp/col_pop/cp.yml')
    config_file = "exp/col_pop/cp.jsonnet"

    overrides = {}
    if "cache_usage" in setting and setting["cache_usage"] == "write":
        overrides["trainer"]["num_epochs"] = 1
    cmd_builder(setting, config_file, json.dumps(overrides))
    main()




