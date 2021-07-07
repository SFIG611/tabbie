
import logging
from typing import Dict, List
from overrides import overrides
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer

from table_embedder.readers.lib.pretrained_transformer_pre_tokenizer import BertPreTokenizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pretrain_reader")
class TablesDatasetReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 tokenizer: PretrainedTransformerTokenizer = None,
                 # tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertPreTokenizer()
        self._token_indexers = token_indexers

    @overrides
    def _read(self, fn: str, max_rows=30, max_cols=20):
        with open(fn, "r") as data_file:
            for line in data_file:
                yield self.text_to_instance(table_id=0)

    def get_table_header_field(self, table_header):
        table_header_field: List[TextField] = []
        for header in table_header:
            tokenized_header = self._tokenizer.tokenize(header)
            table_header_field.append(TextField(tokenized_header, self._token_indexers))
        return ListField(table_header_field)

    def text_to_instance(self, table_id) -> Instance:
        fields = {'table_info': MetadataField({'table_id': table_id,}),}
        table_header_field = self.get_table_header_field(['' for i in range(1)])
        # fields['indexed_headers'] = table_header_field
        # fields['indexed_cells'] = table_header_field
        return Instance(fields)


