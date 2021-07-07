from typing import Dict, Optional
from overrides import overrides

import os
import copy
import time
import json
# import tables
import torch
import numpy as np
from pathlib import Path
# from prettytable import PrettyTable

from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder

from table_embedder.models.embedder_util import TableUtil
# from table_embedder.models.embedder_util import PredUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder

# from scripts.db_reader import TableReader
# torch.set_default_tensor_type(torch.DoubleTensor)


@Model.register("pretrain_dev")
class TableEmbedder(Model):

    def __init__(self, vocab: Vocabulary,
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
        # self.loss = torch.nn.BCELoss()
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "haccuracy": CategoricalAccuracy(),
            "caccuracy": CategoricalAccuracy(),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_max_row_pos, self.num_max_col_pos = 35, 25
        self.opt_level = os.getenv('opt_level')
        self.cls_col = np.load(os.getenv("clscol_path"))
        self.cls_row = np.load(os.getenv("clsrow_path"))

        if os.getenv('emb_path') is not None:
            self.f_pred = open(os.getenv('emb_path'), 'w')

        if not os.getenv('use_cache') is None:
            refresh_db = False if os.getenv('refresh_db') == '-1' else True
            self.tab_db = TableEmbedder.load_db_cache(torch.cuda.current_device(), refresh_db)
        initializer(self)

    @staticmethod
    def load_db_cache(device_id, refresh_db):
        return TableReader(os.getenv('tab_cache{}'.format(device_id)), int(os.getenv('max_bs')), refresh_db=refresh_db)

    # TODO: modify function name
    def pred_by_2d_transformer(self, row_embs, col_embs, label_mask_cls, bs, max_rows, max_cols):
        cells = torch.cat([row_embs, col_embs], dim=3)  # (bs, n_rows, n_cols, 768)
        out_prob_cell = self.feedforward(cells)  # (bs, n_rows, n_cols, 2)
        cell_mask = label_mask_cls.reshape(bs, max_rows, max_cols, 1)
        # out_prob_cell = util.masked_softmax(out_prob_cell, cell_mask_mod)
        out_prob_cell = out_prob_cell * cell_mask.expand_as(out_prob_cell)
        return out_prob_cell

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.metrics["accuracy"].get_metric(reset=reset)
        h_accuracy = self.metrics["haccuracy"].get_metric(reset=reset)
        c_accuracy = self.metrics["caccuracy"].get_metric(reset=reset)
        return {'accuracy': accuracy, 'h_acc': h_accuracy, 'c_acc': c_accuracy}

    def get_tabemb(self, bert_header, bert_data, n_rows, n_cols, bs, table_mask, nrows, ncols):
        print(bert_header.shape, bert_data.shape, n_rows, n_cols, bs, table_mask.shape)
        print(nrows, ncols)
        row_pos_ids = torch.arange(0, self.num_max_row_pos, device=self.device, dtype=torch.long)
        col_pos_ids = torch.arange(0, self.num_max_col_pos, device=self.device, dtype=torch.long)
        print(row_pos_ids.shape, col_pos_ids.shape)

        n_rows_cls = n_rows+1  # row CLS
        n_cols_cls = n_cols+1  # col CLS
        cls_col = torch.from_numpy(copy.deepcopy(self.cls_col)).to(device=self.device)
        cls_row = torch.from_numpy(copy.deepcopy(self.cls_row)).to(device=self.device)
        row_pos_embs = self.row_pos_embedding(row_pos_ids[:n_rows_cls+1])
        col_pos_embs = self.col_pos_embedding(col_pos_ids[:n_cols_cls])

        for i in range(1, 13):
            transformer_row = getattr(self, 'transformer_row{}'.format(str(i)))
            transformer_col = getattr(self, 'transformer_col{}'.format(str(i)))
            if i == 1:
                bert_data = TableUtil.add_cls_tokens(bert_header, bert_data, cls_row, cls_col, bs, n_rows_cls, n_cols_cls)
                bert_data += row_pos_embs.expand((bs, n_cols_cls, n_rows_cls+1, 768)).permute(0, 2, 1, 3).expand_as(bert_data)
                bert_data += col_pos_embs.expand((bs, n_rows_cls+1, n_cols_cls, 768)).expand_as(bert_data)
                table_mask = TableUtil.add_cls_mask(table_mask, bs, n_rows_cls, n_cols_cls, self.device, nrows, ncols)
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows_cls, n_cols_cls, table_mask, transformer_col, self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows_cls, n_cols_cls, table_mask, transformer_row, self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows_cls, n_cols_cls, table_mask, transformer_row, self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows_cls, n_cols_cls, table_mask, transformer_col, self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0
        return row_embs, col_embs, n_rows_cls, n_cols_cls

    def get_labels(self, table_info, bs, max_rows, max_cols):
        cell_labels = TableUtil.get_cell_labels(table_info, bs, max_rows + 1 - 1, max_cols + 1 - 1, self.device)
        header_labels = TableUtil.get_header_labels(table_info, bs, max_cols + 1 - 1, self.device)
        labels = torch.cat([header_labels, cell_labels], dim=1)
        return labels

    def get_mask(self, table_info, bs, max_rows, max_cols):
        table_mask = TableUtil.get_table_mask(table_info, bs, max_rows, max_cols, self.device)
        label_mask = TableUtil.get_table_mask_blank(table_info, table_mask)
        return table_mask, label_mask

    def get_meta(self, table_info):
        nrows = [one_info['num_rows'] for one_info in table_info]
        ncols = [one_info['num_cols'] for one_info in table_info]
        tids = [one_info['id'] for one_info in table_info]
        return nrows, ncols, tids

    @staticmethod
    def to_1d_vec(prob_cells, cell_labels, cell_mask):
        n_cells = np.prod(cell_labels.shape)
        nega_prob_1d = prob_cells[:, :, :, 0].reshape(n_cells)  # (10, 15, 4)
        pos_prob_1d = prob_cells[:, :, :, 1].reshape(n_cells)  # (10, 15, 4)
        all_prob_1d = torch.stack([nega_prob_1d, pos_prob_1d], dim=1)  # (10, 15, 4)
        cell_labels_1d = cell_labels.reshape(n_cells)
        cell_mask_1d = cell_mask.reshape(n_cells)
        return all_prob_1d, cell_labels_1d, cell_mask_1d

    @overrides
    def forward(self, table_info: Dict[str, str],  # ) -> Dict[str, torch.Tensor]:
                indexed_headers: Dict[str, torch.LongTensor], # -> Dict[str, torch.Tensor]:
                indexed_cells: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        t_start = time.time()
        if os.getenv('task_type') != 'pred':
            table_mask, label_mask, labels, table, max_rows, max_cols, tids, nrows, ncols = self.tab_db.load_batch_npy(self.device)
            bs, labels, bert_header, bert_cell = len(table), labels.long(), table[:, 0, :, :], table[:, 1:, :, :]
            print(table_mask.shape, label_mask.shape, labels.shape, table.shape, max_rows, max_cols, tids, nrows, ncols)
            print(bs, labels.shape, bert_header.shape, bert_cell.shape)
            print('\nload batch done: {}, bs: {}'.format(time.time()-t_start, bs))
        else:
            if not hasattr(self, 'bert_embedder'):
                self.bert_embedder = PretrainedBertEmbedder(pretrained_model='bert-base-uncased', top_layer_only=True)
                self.bert_embedder.to(device=self.device)
            self.bert_embedder.eval()
            bs, max_rows, max_cols = TableUtil.get_max_row_col(table_info)
            nrows, ncols, tids = self.get_meta(table_info)
            table_mask, label_mask = self.get_mask(table_info, bs, max_rows, max_cols)
            bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, max_rows, max_cols, None, self.bert_embedder, None, self.device)
            labels = self.get_labels(table_info, bs, max_rows, max_cols)

        # row_embs: (bs, max_rows+2, max_cols+1, n_dim)
        row_embs, col_embs, max_rows_cls, max_cols_cls = self.get_tabemb(bert_header, bert_cell, max_rows, max_cols, bs, table_mask, nrows, ncols)
        label_mask_cls = TableUtil.add_cls_mask(label_mask, bs, max_rows_cls, max_cols_cls, self.device, nrows, ncols)
        prob_tables = self.pred_by_2d_transformer(row_embs, col_embs, label_mask_cls, bs, max_rows_cls+1, max_cols_cls)
        print('bert to prob: {}'.format(time.time()-t_start))

        # eval
        h_prob_1d, h_labels_1d, h_mask_1d = self.to_1d_vec(prob_tables[:, 1, 1:, :].unsqueeze(1), labels[:, 0, :].unsqueeze(1), label_mask_cls[:, 1, 1:].unsqueeze(1))
        c_prob_1d, c_labels_1d, c_mask_1d = self.to_1d_vec(prob_tables[:, 2:, 1:, :], labels[:, 1:, :], label_mask_cls[:, 2:, 1:])
        c_loss = self.loss_func(c_prob_1d[c_mask_1d.bool(), 1], c_labels_1d[c_mask_1d.bool()].float())

        self.metrics['caccuracy'](c_prob_1d, c_labels_1d, mask=c_mask_1d)
        if c_loss != c_loss:  # handle if loss is nan
            c_loss = 0

        # output
        out_dict = {}
        if h_mask_1d.sum() != 0:
            h_loss = self.loss_func(h_prob_1d[h_mask_1d.bool(), 1], h_labels_1d[h_mask_1d.bool()].float())
            self.metrics['haccuracy'](h_prob_1d, h_labels_1d, mask=h_mask_1d)
            print(h_loss, c_loss)
            out_dict['loss'] = h_loss + c_loss
        else:
            out_dict['loss'] = c_loss
        out_dict = self.add_metadata(out_dict, prob_tables, labels, table_info)
        if os.getenv('dump_emb') is not None:
            print('test')
            exit()
            out_dict = self.add_emb(out_dict, row_embs, col_embs)
        # if not self.training:
        #     self.dump_emb(table_info, row_embs, col_embs)
        return out_dict

    @staticmethod
    def add_emb(out_dict, row_embs, col_embs):
        out_dict['row_embs'] = row_embs
        out_dict['col_embs'] = col_embs
        return out_dict

    @staticmethod
    def add_metadata(out_dict, prob_tables, labels, table_info):
        out_dict['prob_headers'] = util.masked_softmax(prob_tables[:, 1, 1:, :], None)
        out_dict['prob_cells'] = util.masked_softmax(prob_tables[:, 2:, 1:, :], None)
        out_dict['label'] = labels

        for one_info in table_info:
            for k, v in one_info.items():
                out_dict[k] = out_dict.get(k, [])
                out_dict[k].append(v)
        return out_dict

    def dump_emb(self, table_info, row_embs, col_embs):
        for k, one_info in enumerate(table_info):
            for i in range(one_info['num_rows']):
                one_row = {}
                one_row['id'] = one_info['id']+'--'+str(i)
                one_row['row_embs'] = torch.autograd.Variable(row_embs[k, i, 0, :].clone(), requires_grad=False).cpu().numpy().tolist()
                one_row['col_embs'] = torch.autograd.Variable(col_embs[k, i, 0, :].clone(), requires_grad=False).cpu().numpy().tolist()
                self.f_pred.write(json.dumps(one_row)+'\n')
        self.f_pred.flush()

