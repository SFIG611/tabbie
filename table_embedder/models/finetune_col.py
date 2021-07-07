
from typing import Dict, Optional
from overrides import overrides

import os
import copy
import torch
import numpy as np
from pathlib import Path
from scripts.to_npy import ToNpy

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
# from allennlp.modules.token_embedders import PretrainedBertEmbedder
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder

from table_embedder.models.embedder_util import TableUtil
# from embedder_util import TableUtil
# from embedder_util import PredUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder
# from lib.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.models.archival import load_archive

from table_embedder.models.cache_util import CacheUtil
# from cache_util import CacheUtil
# torch.set_default_tensor_type(torch.DoubleTensor)


@Model.register("finetune_col")
class TableEmbedder(Model):

    def __init__(self, vocab: Vocabulary,
                 bert_embbeder: PretrainedBertEmbedder,
                 feedforward: FeedForward,
                 row_pos_embedding: Embedding,
                 col_pos_embedding: Embedding,
                 transformer_col1: StackedSelfAttentionEncoder,
                 transformer_col2: StackedSelfAttentionEncoder,
                 transformer_col3: StackedSelfAttentionEncoder,
                 transformer_col4: StackedSelfAttentionEncoder,
                 transformer_col5: StackedSelfAttentionEncoder,
                 transformer_col6: StackedSelfAttentionEncoder,
                 transformer_col7: StackedSelfAttentionEncoder,
                 transformer_col8: StackedSelfAttentionEncoder,
                 transformer_col9: StackedSelfAttentionEncoder,
                 transformer_col10: StackedSelfAttentionEncoder,
                 transformer_col11: StackedSelfAttentionEncoder,
                 transformer_col12: StackedSelfAttentionEncoder,
                 transformer_row1: StackedSelfAttentionEncoder,
                 transformer_row2: StackedSelfAttentionEncoder,
                 transformer_row3: StackedSelfAttentionEncoder,
                 transformer_row4: StackedSelfAttentionEncoder,
                 transformer_row5: StackedSelfAttentionEncoder,
                 transformer_row6: StackedSelfAttentionEncoder,
                 transformer_row7: StackedSelfAttentionEncoder,
                 transformer_row8: StackedSelfAttentionEncoder,
                 transformer_row9: StackedSelfAttentionEncoder,
                 transformer_row10: StackedSelfAttentionEncoder,
                 transformer_row11: StackedSelfAttentionEncoder,
                 transformer_row12: StackedSelfAttentionEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TableEmbedder, self).__init__(vocab, regularizer)
        self.row_pos_embedding = row_pos_embedding
        self.col_pos_embedding = col_pos_embedding
        self.feedforward = feedforward
        self.bert_embedder = bert_embbeder

        # self.transformer_col = transformer_col
        self.transformer_col1 = transformer_col1
        self.transformer_col2 = transformer_col2
        self.transformer_col3 = transformer_col3
        self.transformer_col4 = transformer_col4
        self.transformer_col5 = transformer_col5
        self.transformer_col6 = transformer_col6
        self.transformer_col7 = transformer_col7
        self.transformer_col8 = transformer_col8
        self.transformer_col9 = transformer_col9
        self.transformer_col10 = transformer_col10
        self.transformer_col11 = transformer_col11
        self.transformer_col12 = transformer_col12
        # self.transformer_row = transformer_row
        self.transformer_row1 = transformer_row1
        self.transformer_row2 = transformer_row2
        self.transformer_row3 = transformer_row3
        self.transformer_row4 = transformer_row4
        self.transformer_row5 = transformer_row5
        self.transformer_row6 = transformer_row6
        self.transformer_row7 = transformer_row7
        self.transformer_row8 = transformer_row8
        self.transformer_row9 = transformer_row9
        self.transformer_row10 = transformer_row10
        self.transformer_row11 = transformer_row11
        self.transformer_row12 = transformer_row12
        self.loss = torch.nn.BCELoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "haccuracy": CategoricalAccuracy(),
            "caccuracy": CategoricalAccuracy(),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_func = torch.nn.BCELoss()

        self.num_max_row_pos = 35
        self.num_max_col_pos = 25

        self.cache_usage = os.getenv("cache_usage")
        self.cache_dir = os.getenv('cache_dir')
        self.cls_col = np.load(os.getenv("clscol_path"))
        self.cls_row = np.load(os.getenv("clsrow_path"))

        if self.cache_dir is not None:
            cache_dir = Path(self.cache_dir)
            self.cell_feats, self.cell_id = ToNpy.load_cid(cache_dir/'cell_feats.npy', cache_dir/'cell_id.txt')

        if os.getenv('model_path') is not None and os.getenv('learn_type') != 'pred':
            self.init_weight()

        self.opt_level = 'O0'
        initializer(self)

    def init_weight(self):
        model_path = os.getenv('model_path')
        archive = load_archive(model_path)
        # https://github.com/allenai/allennlp-semparse/blob/master/allennlp_semparse/models/wikitables/wikitables_erm_semantic_parser.py
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                new_weights = weights.data
                model_parameters[name].data.copy_(new_weights)

    def pred_by_2d_transformer(self, row_embs, col_embs, table_mask_cls, bs, max_rows, max_cols):
        cells = torch.cat([row_embs, col_embs], dim=3)  # (10, 15, 4, 1536)
        out_prob_cell = self.feedforward(cells)  # (10, 15, 4, 2)
        cell_mask_mod = table_mask_cls.reshape(bs, max_rows, max_cols, 1)
        out_prob_cell = util.masked_softmax(out_prob_cell, cell_mask_mod)  # (10, 2)
        return out_prob_cell

    def get_labels(self, table_info, bs, n_rows, n_cols):
        header_labels = torch.zeros((bs, n_cols), device=self.device)
        cell_labels = torch.zeros((bs, n_rows, n_cols), device=self.device)
        for k, one_info in enumerate(table_info):
            if 'col_labels' in one_info and one_info['col_labels'] is not None:
                for label_idx in one_info['col_labels']:
                    header_labels[k][label_idx] = 1
            if 'cell_labels' in one_info and one_info['cell_labels'] is not None:
                for label_idx in one_info['cell_labels']:
                    cell_labels[k][label_idx[0]][label_idx[1]] = 1
        return header_labels, cell_labels

    def get_tabemb(self, bert_header, bert_data, n_rows, n_cols, bs, table_mask, nrows, ncols):
        row_pos_ids = torch.arange(0, self.num_max_row_pos, device=self.device, dtype=torch.long)
        col_pos_ids = torch.arange(0, self.num_max_col_pos, device=self.device, dtype=torch.long)

        n_rows += 1  # row CLS
        n_cols += 1  # col CLS
        cls_col = torch.from_numpy(copy.deepcopy(self.cls_col)).to(device=self.device)
        cls_row = torch.from_numpy(copy.deepcopy(self.cls_row)).to(device=self.device)
        row_pos_embs = self.row_pos_embedding(row_pos_ids[:n_rows+1])
        col_pos_embs = self.col_pos_embedding(col_pos_ids[:n_cols])

        for i in range(1, 13):
            transformer_row = getattr(self, 'transformer_row{}'.format(str(i)))
            transformer_col = getattr(self, 'transformer_col{}'.format(str(i)))
            if i == 1:
                bert_data = TableUtil.add_cls_tokens(bert_header, bert_data, cls_row, cls_col, bs, n_rows, n_cols)
                bert_data += row_pos_embs.expand((bs, n_cols, n_rows + 1, 768)).permute(0, 2, 1, 3).expand_as(bert_data)
                bert_data += col_pos_embs.expand((bs, n_rows + 1, n_cols, 768)).expand_as(bert_data)
                table_mask_cls = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask_cls, transformer_col, self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask_cls, transformer_row, self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask_cls, transformer_row, self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask_cls, transformer_col, self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0
        return row_embs, col_embs, n_rows, n_cols, table_mask_cls

    # def get_tab_emb(self, bert_header, bert_data, n_rows, n_cols, table_info, bs, table_mask):
    #     row_pos_ids = torch.arange(0, self.num_max_row_pos, device=self.device, dtype=torch.long)
    #     col_pos_ids = torch.arange(0, self.num_max_col_pos, device=self.device, dtype=torch.long)

    #     n_rows += 1  # row CLS
    #     n_cols += 1  # col CLS
    #     cls_col = torch.from_numpy(copy.deepcopy(self.cls_col)).to(device=self.device)
    #     cls_row = torch.from_numpy(copy.deepcopy(self.cls_row)).to(device=self.device)
    #     row_pos_embs = self.row_pos_embedding(row_pos_ids[:(n_rows+1)])  # +1 for header
    #     col_pos_embs = self.col_pos_embedding(col_pos_ids[:n_cols])

    #     for i in range(1, 13):
    #         transformer_row = getattr(self, 'transformer_row{}'.format(str(i)))
    #         transformer_col = getattr(self, 'transformer_col{}'.format(str(i)))
    #         if i == 1:
    #             bert_data = TableUtil.add_cls_tokens(bert_header, bert_data, cls_row, cls_col, bs, n_rows, n_cols)
    #             bert_data += row_pos_embs.expand((bs, n_cols, n_rows + 1, 768)).permute(0, 2, 1, 3).expand_as(bert_data)
    #             bert_data += col_pos_embs.expand((bs, n_rows + 1, n_cols, 768)).expand_as(bert_data)
    #             table_mask_cls = TableUtil.add_cls_mask(table_mask, table_info, bs, n_rows, n_cols, self.device)
    #             col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask_cls, transformer_col)
    #             row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask_cls, transformer_row)
    #         else:
    #             row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask_cls, transformer_row)
    #             col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask_cls, transformer_col)
    #         ave_embs = (row_embs + col_embs) / 2.0
    #     return row_embs, col_embs, n_rows, n_cols, table_mask_cls

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.metrics["accuracy"].get_metric(reset=reset)
        # h_accuracy = self.metrics["haccuracy"].get_metric(reset=reset)
        # c_accuracy = self.metrics["caccuracy"].get_metric(reset=reset)
        return {'accuracy': accuracy}
        # return {'accuracy': accuracy, 'h_acc': h_accuracy, 'c_acc': c_accuracy}

    def get_meta(self, table_info):
        nrows = [one_info['num_rows'] for one_info in table_info]
        ncols = [one_info['num_cols'] for one_info in table_info]
        tids = [one_info['table_id'] for one_info in table_info]
        return nrows, ncols, tids

    @overrides
    def forward(self, table_info: Dict[str, str],
                indexed_headers: Dict[str, torch.LongTensor],#) -> Dict[str, torch.Tensor]:
                indexed_cells: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        # initialize
        self.bert_embedder.eval()
        bs, n_rows, n_cols = TableUtil.get_max_row_col(table_info)
        nrows, ncols, tids = self.get_meta(table_info)
        table_mask = TableUtil.get_table_mask(table_info, bs, n_rows, n_cols, self.device)

        # to row/col emb
        if self.cache_dir is None:
            bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, n_rows, n_cols, self.cache_usage, self.bert_embedder, None, self.device)
        else:
            bert_header, bert_cell = TableUtil.to_bert_emb(table_info, bs, n_rows, n_cols, self.device, self.cell_id, self.cell_feats)
        # row_embs_cls, col_embs_cls, n_rows_cls, n_cols_cls, table_mask_cls = self.get_tab_emb(bert_header, bert_cell, n_rows, n_cols, table_info, bs, table_mask)
        row_embs, col_embs, n_rows_cls, n_cols_cls, table_mask_cls = self.get_tabemb(bert_header, bert_cell, n_rows, n_cols, bs, table_mask, nrows, ncols)
        # table_mask = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
        # print(row_embs.shape, col_embs.shape, table_mask.shape, bs, n_rows_cls, n_cols_cls)

        # to fake cell prob
        prob_tables_cls = self.pred_by_2d_transformer(row_embs, col_embs, table_mask_cls, bs, n_rows_cls+1, n_cols_cls)
        prob_headers, prob_cells = prob_tables_cls[:, 1, 1:, :], prob_tables_cls[:, 2:, 1:, :]  # (bs, n_rows, n_cols, 768)

        # get labels
        header_labels, cell_labels = self.get_labels(table_info, bs, n_rows, n_cols)  # TODO: modify
        labels = torch.cat((header_labels.reshape(bs, -1, header_labels.shape[-1]), cell_labels), axis=1)
        header_labels_1d, cell_labels_1d = header_labels.reshape(-1), cell_labels.reshape(-1)
        header_mask_1d, cell_mask_1d = table_mask[:, 0, :].reshape(-1), table_mask[:, 1:, :].reshape(-1)
        prob_headers_pos, prob_cells_pos, prob_headers_nega, prob_cells_nega = prob_headers[:, :, 1], prob_cells[:, :, :, 1], prob_headers[:, :, 0], prob_cells[:, :, :, 0]
        prob_headers_pos_1d, prob_cells_pos_1d, prob_headers_nega_1d, prob_cells_nega_1d = prob_headers_pos.reshape(-1), prob_cells_pos.reshape(-1), prob_headers_nega.reshape(-1), prob_cells_nega.reshape(-1)

        # cell_loss = self.loss_func(prob_cells_pos_1d[cell_mask_1d.bool()], cell_labels_1d[cell_mask_1d.bool()].float())
        header_loss = self.loss_func(prob_headers_pos_1d[header_mask_1d.bool()], header_labels_1d[header_mask_1d.bool()].float())
        # print(prob_cells_nega_1d.shape, prob_cells_pos_1d.shape, cell_mask_1d.shape)
        # prob_cells = torch.stack([prob_cells_nega_1d, prob_cells_pos_1d], dim=1)
        # prob_cells = torch.stack([prob_cells_nega_1d, prob_cells_pos_1d], dim=1)
        # self.metrics['accuracy'](prob_cells, cell_labels_1d, mask=cell_mask_1d)
        # self.metrics['accuracy'](prob_headers, header_labels_1d, mask=header_mask_1d)

        cell_loss = self.loss_func(prob_cells_pos_1d[cell_mask_1d.bool()], cell_labels_1d[cell_mask_1d.bool()].float())
        prob_headers_acc = torch.stack([prob_headers_nega_1d, prob_headers_pos_1d], dim=1)
        prob_cells_acc = torch.stack([prob_cells_nega_1d, prob_cells_pos_1d], dim=1)
        # self.metrics['accuracy'](prob_cells, cell_labels_1d, mask=cell_mask_1d)
        self.metrics['accuracy'](prob_headers_acc, header_labels_1d, mask=header_mask_1d)

        # h_loss_weight = 0.0
        # loss = cell_loss * (1.0-h_loss_weight) + header_loss * h_loss_weight
        output_dict = {'loss': header_loss}
        if not self.training:
            output_dict = self.add_metadata(table_info, output_dict)
            # output_dict['prob_headers'] = prob_headers_pos
            # output_dict['prob_cells'] = prob_cells_pos
            output_dict['prob_headers'] = prob_headers
            output_dict['prob_cells'] = prob_cells
        return output_dict

    @staticmethod
    def add_metadata(table_info, output_dict):
        data_dict = {}
        for one_info in table_info:
            if 'table_id' in one_info:
                one_info['id'] = one_info['table_id']
            for k, v in one_info.items():
                data_dict[k] = data_dict.get(k, [])
                data_dict[k].append(v)
        output_dict.update(data_dict)
        return output_dict


