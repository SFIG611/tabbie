
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from scripts.util import Util
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField, ListField
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.batch import Batch


class BertUtil:
    def __init__(self, is_eval):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PretrainedTransformerTokenizer(model_name="bert-base-uncased")
        self.token_indexers = {
            'bert': PretrainedTransformerIndexer(model_name="bert-base-uncased")
        }
        self.bert_embedder = PretrainedBertEmbedder(pretrained_model='bert-base-uncased', top_layer_only=True)
        self.bert_embedder.to(self.device)
        if is_eval:
            self.bert_embedder.eval()

    def to_field(self, cells):
        cell_field: List[TextField] = []
        for cell in cells:
            tokenized_cell = self.tokenizer.tokenize(cell)
            # https://github.com/allenai/allennlp/issues/1887
            # tokenized_header = [Token(token.text) for token in tokenized_header]
            cell_field.append(TextField(tokenized_cell, self.token_indexers))
        return ListField(cell_field)

    def to_token_ids(self, cells):
        vocab = Vocabulary()
        cell_field = self.to_field(cells)
        instance = Instance({'cell': cell_field})  # check this code later
        instance.index_fields(vocab)
        data = Batch([instance]).as_tensor_dict()
        token_ids = data['cell']['bert']['token_ids']
        return token_ids

    def dump_emb(self, jsonl_path, out_dir, bs=20, feat_dim=768):
        cells = Util.get_uniq_cells(jsonl_path)
        n_cells, pad_len = len(cells), bs-len(cells)%bs
        cells_batches = np.array(cells + ['' for i in range(pad_len)]).reshape(-1, bs)

        bert_emb = np.empty((0, feat_dim), dtype=np.float32)
        for cells_batch in cells_batches:
            token_ids = self.to_token_ids(cells_batch)
            token_ids = token_ids.to(self.device, dtype=torch.long)
            emb_batch = self.bert_embedder(token_ids)[0, :, 0, :]
            emb_batch = torch.autograd.Variable(emb_batch.clone(), requires_grad=False).detach().cpu().numpy()
            emb_batch = emb_batch.reshape(-1, feat_dim)
            bert_emb = np.vstack([bert_emb, emb_batch])
        np.save(out_dir/'cell_feats.npy', bert_emb[:n_cells])
        Util.dump_list(cells[:n_cells], out_dir/'cell_id.txt')


def main():
    device = 1
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    os.environ["ALLENNLP_DEBUG"] = "TRUE"
    bert_util = BertUtil(device, True)
    bert_util.dump_emb('../data/ft_cell/train.jsonl', Path('../data/ft_cell/'))


if __name__ == '__main__':
    main()


