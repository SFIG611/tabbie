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
# from table_embedder.dataset_readers.lib.pretrained_transformer_pre_tokenizer import BertPreTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sato_reader")
class TablesDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 tokenizer: PretrainedTransformerTokenizer = None,
                 # tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
                 # tokenizer: BertPreTokenizer = None,
                 # token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertPreTokenizer()
        self._token_indexers = token_indexers
        # self._tokenizer = tokenizer or BertPreTokenizer()
        # self._token_indexers = token_indexers or {"bert": PretrainedBertIndexer(os.getenv("bert_vocab_path"))}

    @staticmethod
    def parse_table(table, max_rows, max_cols):
        header, cells = table[0], table[1:max_rows]
        if len(header) > max_cols:
            header = np.array(header)[:max_cols].tolist()
            cells = np.array(cells)[:, :max_cols].tolist()
        return header, cells

    @overrides
    def _read(self, fn: str, max_rows=30, max_cols=20):
        with open(fn, "r") as data_file:
            for line in data_file:
                line = json.loads(line)
                table = line['table_data']

                header = ['' for i in range(len(table[0]))]
                table = np.vstack([header, table]).tolist()
                header, cells = self.parse_table(table, max_rows, max_cols)

                # for elem in line['col_idx']:
                #     print(elem, line['col_idx'])

                col_idx = []
                for elem in line['col_idx']:
                    if int(elem) < 20 and int(elem) < len(header):
                        col_idx.append(elem)

                # if len(label_idx) != len(col_idx):
                #     print(label_idx, col_idx)

                if len(cells) >= 1:
                    instance = self.text_to_instance(
                        table_id=line['id'],
                        table_header=header,
                        table_data=cells,
                        n_rows=len(cells),
                        n_cols=len(header),
                        fname=data_file.name,
                        label_idx=line['label_idx'],
                        col_idx=col_idx,
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

    def get_table_data_field(self, table_data):
        table_data_field: List[ListField] = []
        for row in table_data:
            row_field: List[TextField] = []
            for cell in row:
                tokenized_cell = self._tokenizer.tokenize(cell[:300])
                # tokenized_cell = [Token(token.text) for token in tokenized_cell]
                row_field.append(TextField(tokenized_cell, self._token_indexers))
            table_data_field.append(ListField(row_field))
        return ListField(table_data_field)

    def text_to_instance(self, table_id, table_header, table_data, n_rows, n_cols, fname, label_idx, col_idx) -> Instance:
        fields = {
            'table_info': MetadataField({
                'id': table_id,
                'fname': fname,
                'num_rows': n_rows,
                'num_cols': n_cols,
                'header': table_header,
                'table_data_raw': table_data,
                'col_idx': col_idx,
                'label_idx': label_idx})
        }

        table_header_field = self.get_table_header_field(table_header)
        fields['indexed_headers'] = table_header_field
        # fields['indexed_cells'] = self.get_table_data_field(table_data)
        return Instance(fields)

# header, cells = table[0], table[1:max_rows]
# if len(header) > max_cols:
#     header = np.array(header)[:max_cols].tolist()
#     cells = np.array(cells)[:, :max_cols].tolist()


