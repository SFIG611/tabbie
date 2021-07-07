
from typing import Dict, List
from overrides import overrides

import os
import json
import logging
import numpy as np

from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
import copy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("preprocess")
class TablesDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 tokenizer: PretrainedTransformerTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer # or BertPreTokenizer()
        self._token_indexers = token_indexers  # or {"bert": PretrainedBertIndexer(os.getenv("bert_vocab_path"))}
        self.max_cell_len = int(os.getenv('max_cell_len'))

    @overrides
    def _read(self, fn: str, max_rows=30, max_cols=20):
        with open(fn, "r") as data_file:
            for line in data_file:
                line = json.loads(line)
                table = line['table_data']
                if len(table[0]) > max_cols:
                    table = np.array(table)[:, :max_cols].tolist()

                if len(table) >= 1:
                    instance = self.text_to_instance(
                        id=line['id'],
                        header=table[0],
                        table=table[1:max_rows],
                    )
                    if instance is not None:
                        yield instance

    def get_table_header_field(self, table_header):
        table_header_field: List[TextField] = []
        for header in table_header:
            tokenized_header = self._tokenizer.tokenize(header[:self.max_cell_len])
            # https://github.com/allenai/allennlp/issues/1887
            # tokenized_header = [Token(token.text) for token in tokenized_header]
            table_header_field.append(TextField(tokenized_header, self._token_indexers))
        return ListField(table_header_field)

    def get_table_data_field(self, table_data):
        table_data_field: List[ListField] = []
        for row in table_data:
            row_field: List[TextField] = []
            for cell in row:
                tokenized_cell = self._tokenizer.tokenize(cell[:self.max_cell_len])
                # tokenized_cell = [Token(token.text) for token in tokenized_cell]
                row_field.append(TextField(tokenized_cell, self._token_indexers))
            table_data_field.append(ListField(row_field))
        return ListField(table_data_field)

    @staticmethod
    def to_fake_table(header, table, fake_cell_ids, fake_cell_words, fake_headers):
        if fake_headers is not None:
            for elem in fake_headers:
                header[elem[0]] = elem[1]
        if fake_cell_ids is not None:
            for k, elem in enumerate(fake_cell_ids):
                table[elem[0]][elem[1]] = fake_cell_words[k]
        return header, table

    def text_to_instance(self, id, header, table, blank_loc=None, replace_cell_ids=None, replace_cell_words=None, replace_headers=None) -> Instance:
        fields = {
            'table_info': MetadataField({
                'id': id,
                'num_rows': len(table),
                'num_cols': len(header),
                'header': copy.deepcopy(header),
                'blank_loc': blank_loc,
                'replace_cell_ids': replace_cell_ids,
                'replace_cell_words': replace_cell_words,
                'replace_headers': replace_headers,
                'table_data_raw': copy.deepcopy(table)}),
        }
        header, table = self.to_fake_table(header, table, replace_cell_ids, replace_cell_words, replace_headers)
        fields['indexed_headers'] = self.get_table_header_field(header)
        fields['indexed_cells'] = self.get_table_data_field(table)
        return Instance(fields)



