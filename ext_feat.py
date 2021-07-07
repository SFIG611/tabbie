import sys
import os
import argparse
import yaml
import json
import numpy as np

from pathlib import Path
from scripts.util import Util
# from allennlp.service.predictors import Predictor
from allennlp.predictors import Predictor
from table_embedder.predictors.predictor import PretrainPredictor
from allennlp.models.archival import load_archive
from table_embedder.models.pretrain import TableEmbedder
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from table_embedder.readers.preprocess import TablesDatasetReader


class ExtFeat:
    def __init__(self, model_path):
        self.model = ExtFeat.load_model(model_path)
        self.reader = TablesDatasetReader(tokenizer=PretrainedTransformerTokenizer(model_name='bert-base-uncased'),
                                           token_indexers={'bert': PretrainedTransformerIndexer(model_name='bert-base-uncased')})
        self.predictor = PretrainPredictor(self.model, dataset_reader=self.reader)
        print('init done')

    @staticmethod
    def load_model(model_path, device=0):
        overrides = json.dumps({'dataset_reader': {'type': 'preprocess'}, 'trainer': {'opt_level': 'O0'}})
        archive = load_archive(model_path, overrides=overrides)
        model = archive.model
        model = model.cuda(device)
        model.eval()
        print('load model done')
        return model

    def dump_feat(self, jsonl_path, out_dir, add_cls=False):
        lines = Util.load_lines(jsonl_path)
        if len(lines) > 1000:
            raise ValueError('too many files')
        for line in lines:
            output = self.predictor.predict_json(line)
            row_embs, col_embs = np.array(output['row_embs']), np.array(output['col_embs'])
            if not add_cls:
                row_embs, col_embs = row_embs[1:, 1:, :], col_embs[1:, 1:, :]
            embs = np.concatenate((row_embs, col_embs), axis=2)
            np.save(out_dir/'{}.npy'.format(line['id']), embs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="fixed params for config", default="./exp/ext_feat/ext_feat.yml")
    parser.add_argument("--csv_dir", help="input csv dir", default="./data/ft_cell/train_csv")
    parser.add_argument("--model_path", help="model file path", default="./model/freq.tar.gz")
    parser.add_argument("--out_dir", help="output npy dir", default="./out_npy")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    tables_path = Path(args.out_dir)/'tables.jsonl'

    Util.csvdir_to_jsonl(Path(args.csv_dir), tables_path, label_path=None)
    params = Util.load_yaml(args.config_path)
    for name, val in params.items():
        if not isinstance(val, str):
            continue
        print(name)
        os.environ[name] = val

    ext_feat = ExtFeat(args.model_path)
    ext_feat.dump_feat(tables_path, out_dir)


