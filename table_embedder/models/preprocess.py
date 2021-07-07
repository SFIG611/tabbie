from typing import Dict, Optional
from overrides import overrides

import os
import torch
from pathlib import Path

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder

from table_embedder.models.embedder_util import TableUtil
from table_embedder.models.cache_util import CacheUtil


@Model.register("preprocess")
class TableEmbedder(Model):

    def __init__(self, vocab: Vocabulary,
                 bert_embbeder: PretrainedBertEmbedder,
                 feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TableEmbedder, self).__init__(vocab, regularizer)
        self.feedforward = feedforward
        self.bert_embedder = bert_embbeder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt_level = 'O0'

        device_id = torch.cuda.current_device()
        cell_db_path = str(Path(os.getenv('out_dir')) / 'cell_db_part{}.kch'.format(device_id))

        if os.path.exists(cell_db_path):
            pass
        else:
            self.cache_util = CacheUtil('write', cell_db_path)

        initializer(self)

    @overrides
    def forward(self, table_info: Dict[str, str],
                indexed_headers: Dict[str, torch.LongTensor],
                indexed_cells: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        # initialize
        self.bert_embedder.eval()
        bs, max_rows, max_cols = TableUtil.get_max_row_col(table_info)

        _, _ = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, max_rows, max_cols, 'write', self.bert_embedder, self.cache_util, self.device)

        # output dummy
        dummy1 = torch.randn(3, 2, dtype=torch.float, device=self.device)
        prob = util.masked_softmax(self.feedforward(dummy1), None)  # (10, 2)
        dummy_bool = torch.cuda.FloatTensor(3).uniform_() > 0.8
        dummy2 = torch.tensor(dummy_bool, dtype=torch.float, device=self.device)

        loss_func = torch.nn.BCELoss()
        loss = loss_func(prob[:, 1], dummy2)
        output_dict = {'loss': loss}

        return output_dict



