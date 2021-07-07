from typing import Dict, List
from overrides import overrides

import os
import time
import json
import copy
import random
import logging
import numpy as np

from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField
from allennlp.data.tokenizers import Tokenizer

# from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
# from table_embedder.dataset_readers.lib.pretrained_transformer_pre_tokenizer import BertPreTokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

from table_embedder.readers.lib.pretrained_transformer_pre_tokenizer import BertPreTokenizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("cell_reader")
class TablesDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 # tokenizer: BertPreTokenizer = None,
                 tokenizer: PretrainedTransformerTokenizer = None,
                 # tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer  # or BertPreTokenizer()
        self._token_indexers = token_indexers  # or {"bert": PretrainedBertIndexer(os.getenv("bert_vocab_path"))}
        self.cache_usage = os.getenv("cache_usage")

    @overrides
    def _read(self, fn: str, max_rows=30, max_cols=20):
        with open(fn, "r") as data_file:
            for line in data_file:
                line = json.loads(line)
                idx = line['old_id'] if 'old_id' in line.keys() else line['id']

                table_data = line['table_data']
                table = copy.deepcopy(table_data)
                table_header = table_data[0]
                table_data = table_data[1:max_rows]
                if len(table_data[0]) > max_cols:
                    table_header = np.array(table_header)[:max_cols].tolist()
                    table_data = np.array(table_data)[:, :max_cols].tolist()

                cell_labels = line['cell_labels'] if 'cell_labels' in line else None
                col_labels = line['col_labels'] if 'col_labels' in line else None
                table_labels = line['table_labels'] if 'table_labels' in line else None

                if len(table_data) >= 1:
                    instance = self.text_to_instance(
                        table_id=idx,
                        header=table_header,
                        # header_labels=line['header_labels'],
                        cell_labels=cell_labels,
                        col_labels=col_labels,
                        table_labels=table_labels,
                        table_data=table_data,
                        table=table
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
                tokenized_cell = self._tokenizer.tokenize(cell)
                # tokenized_cell = [Token(token.text) for token in tokenized_cell]
                row_field.append(TextField(tokenized_cell, self._token_indexers))
            table_data_field.append(ListField(row_field))
        return ListField(table_data_field)

    def text_to_instance(self, table_id, header, cell_labels, col_labels, table_labels, table_data, table) -> Instance:
        fields = {
            'table_info': MetadataField({
                'table_id': table_id,
                'num_rows': len(table_data),
                'num_cols': len(header),
                'header': header,
                # 'header_labels': header_labels,
                'cell_labels': cell_labels,
                'col_labels': col_labels,
                'table_labels': table_labels,
                'table_data_raw': table_data,
                'table': table}),
        }

        table_header_field = self.get_table_header_field(header)
        fields['indexed_headers'] = table_header_field

        if os.getenv('cache_dir') is not None:
            fields['indexed_cells'] = table_header_field  # TODO: delete if needed
        else:
            fields["indexed_cells"] = self.get_table_data_field(table_data)
            fields["indexed_headers"] = self.get_table_header_field(header)
        return Instance(fields)


