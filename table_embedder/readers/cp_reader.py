from typing import Dict, List
from overrides import overrides

import os
import json
import copy
import logging
import numpy as np

from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from table_embedder.readers.lib.pretrained_transformer_pre_tokenizer import BertPreTokenizer

# from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
# f#rom table_embedder.dataset_readers.lib.pretrained_transformer_pre_tokenizer import BertPreTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("cp_reader")
class TablesDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 tokenizer: PretrainedTransformerTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertPreTokenizer()
        self._token_indexers = token_indexers
        # self._tokenizer = tokenizer or BertPreTokenizer()
        # self._token_indexers = token_indexers or {"bert": PretrainedBertIndexer(os.getenv("bert_vocab_path"))}

    @overrides
    def _read(self, fn: str, max_rows=30, max_cols=20):
        with open(fn, "r") as data_file:
            for line in data_file:
                json_data = json.loads(line)

                # prepro table_data
                table_data = json_data['table_data']
                table_header = table_data[0]
                table_data = table_data[1:max_rows]
                if len(table_header) > max_cols:
                    table_header = np.array(table_header)[:max_cols].tolist()
                    table_data = np.array(table_data)[:, :max_cols].tolist()

                # prepro labels
                max_label_idx = 127656
                for k, elem in enumerate(json_data['label_idx']):
                    if int(elem) >= max_label_idx:
                        json_data['label_idx'][k] = 0
                        json_data['label'][k] = ''

                if len(table_data) >= 1:
                    instance = self.text_to_instance(
                        table_id=json_data['id'],
                        table_header=table_header,
                        table_data=table_data,
                        n_rows=len(table_data),
                        n_cols=len(table_header),
                        fname=data_file.name,
                        label_idx=json_data['label_idx'],
                        orig_header=json_data['orig_header'],
                        label=json_data['label'],
                    )
                    if instance is not None:
                        yield instance

    def get_table_header_field(self, table_header):
        table_header_field: List[TextField] = []
        for header in table_header:
            tokenized_header = self._tokenizer.tokenize(header)
            # https://github.com/allenai/allennlp/issues/1887
            # tokenized_header = [Token(token.text) for token in tokenized_header]
            table_header_field.append(TextField(tokenized_header, self._token_indexers))
        return ListField(table_header_field)

    def text_to_instance(self, table_id, table_header, table_data, n_rows, n_cols, fname, label_idx, orig_header, label) -> Instance:
        fields = {
            'table_info': MetadataField({
                'id': table_id,
                'fname': fname,
                'num_rows': n_rows,
                'num_cols': n_cols,
                'header': table_header,
                'table_data_raw': table_data,
                'orig_header': orig_header,
                'label': label,
                'label_idx': label_idx})
        }

        table_header_field = self.get_table_header_field(table_header)
        fields['indexed_headers'] = table_header_field
        return Instance(fields)


