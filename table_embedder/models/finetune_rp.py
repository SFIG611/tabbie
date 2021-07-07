from typing import Dict, Optional
from overrides import overrides

import os
import copy
import time
import torch
import random
import numpy as np
import pandas as pd
from scripts.row_pop import RowPop

from torch.autograd import Variable
from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
# from allennlp.modules.token_embedders import PretrainedBertEmbedder
from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder
from allennlp.models.archival import load_archive

from embedder_util import TableUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder
# from lib.stacked_self_attention import StackedSelfAttentionEncoder

from cache_util import CacheUtil
from .lib.masked_ff import MaskedFeedForward
# torch.set_default_tensor_type(torch.DoubleTensor)


@Model.register("finetune_rp")
class TableEmbedder(Model):

    def __init__(self, vocab: Vocabulary,
                 bert_embbeder: PretrainedBertEmbedder,
                 feedforward: FeedForward,
                 compose_ff: FeedForward,
                 row_pos_embedding: Embedding,
                 col_pos_embedding: Embedding,
                 top_feedforward: MaskedFeedForward,
                 # row_feedforward: FeedForward,
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
        self.top_feedforward = top_feedforward
        # self.row_feedforward = row_feedforward
        self.feedforward = feedforward
        self.compose_ff = compose_ff
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
            "accuracy": CategoricalAccuracy(tie_break=True),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_func = torch.nn.CrossEntropyLoss()

        self.num_max_row_pos = 35
        self.num_max_col_pos = 25
        self.n_classes = int(os.getenv('n_classes'))
        self.n_seed_rows = int(os.getenv('n_seed_rows'))

        self.cache_usage = os.getenv("cache_usage")
        self.cls_col = np.load(os.getenv("clscol_path"))
        self.cls_row = np.load(os.getenv("clsrow_path"))
        self.label = RowPop.load_label(os.getenv('label_path'), key='index')
        self.opt_level = 'O0'

        if self.cache_usage is not None:
            self.cache_util = CacheUtil(self.cache_usage, os.getenv("cell_db_path"))
        else:
            self.cache_util = None

        if os.getenv('model_path') is not None and os.getenv('learn_type') != 'pred':
            self.init_weight()

        initializer(self)

    def init_weight(self):
        model_path = os.getenv('model_path')
        print(model_path)
        archive = load_archive(model_path)
        # https://github.com/allenai/allennlp-semparse/blob/master/allennlp_semparse/models/wikitables/wikitables_erm_semantic_parser.py
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                new_weights = weights.data
                model_parameters[name].data.copy_(new_weights)

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
                # table_mask = TableUtil.add_cls_mask(table_mask, table_info, bs, n_rows, n_cols, self.device)
                table_mask = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_col, self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_row, self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_row, self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_col, self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0
        return row_embs, col_embs, n_rows, n_cols

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.metrics["accuracy"].get_metric(reset=reset)
        return {'accuracy': accuracy}

    @staticmethod
    def get_labels(table_info):
        labels = []
        labels_1d = []
        for one_info in table_info:
            row_labels = list(set(copy.deepcopy(one_info['label_idx'])))
            if 0 in row_labels:
                row_labels.remove(0)
            labels.append(row_labels)
            for elem in row_labels:
                labels_1d.append(elem)
        return labels_1d, labels

    def pred_prob(self, cell_embs, labels, mask):
        out_prob = self.top_feedforward(cell_embs, mask)
        out_prob_1d = []
        for k, one_prob in enumerate(out_prob):
            # out_prob_1d.append(one_prob.expand(len(labels[k]), self.n_classes))
            out_prob_1d.append(one_prob.expand(len(labels[k]), out_prob.shape[1]))
        out_prob_1d = torch.cat(out_prob_1d, dim=0)
        return out_prob_1d, out_prob

    @staticmethod
    def mod_out_prob(out_prob_1d, pred_labels):
        out_prob_cp = torch.autograd.Variable(out_prob_1d.clone(), requires_grad=False).cpu()
        cnt = 0
        for pred_row_labels in pred_labels:
            n_row_label = len(pred_row_labels)
            for i in range(n_row_label):
                for elem in pred_row_labels:
                    out_prob_cp[cnt+i][elem] = torch.max(out_prob_cp[cnt+i])
            cnt += n_row_label
        return out_prob_cp

    @staticmethod
    def get_ave_cls(row_embs, col_embs):
        cls_embs = torch.cat([row_embs[:, 0, 1:, :], col_embs[:, 0, 1:, :]], dim=2)
        # cls_embs = torch.cat([cls_embs[:, 0, :], cls_embs[:, 1, :], cls_embs[:, 2, :]], dim=1)
        cls_embs = cls_embs.mean(dim=1)
        return cls_embs

    @staticmethod
    def cat_cell_embs(row_embs, col_embs, n_seed_rows, device):
        ave_embs = (row_embs + col_embs) / 2.0
        bs = ave_embs.shape[0]
        n_rows = row_embs.shape[1]-2

        col0_embs = ave_embs[:, 2:, 1, :]
        cell_embs = col0_embs[:, 0, :]
        for i in range(1, min(3, n_seed_rows)):
            if n_rows <= i:
                zeros = torch.zeros((bs, 768), device=device)
                cell_embs = torch.cat([cell_embs, zeros], dim=1)
            else:
                cell_embs = torch.cat([cell_embs, col0_embs[:, i, :]], dim=1)
        return cell_embs

    @staticmethod
    def mask_r1_2_embs(cls_embs, table_info):
        for k, one_info in enumerate(table_info):
            if one_info['num_rows'] == 1:
                cls_embs[k, 768:] = 0
        return cls_embs

    @staticmethod
    def mask_r1_3_embs(cls_embs, table_info):
        for k, one_info in enumerate(table_info):
            if one_info['num_rows'] == 1:
                cls_embs[k, 768:] = 0
            elif one_info['num_rows'] == 2:
                cls_embs[k, (768*2):] = 0
        return cls_embs

    @staticmethod
    def add_metadata(table_info, output_dict, pred_labels, pred_labels_name):
        data_dict = {'pred_labels': pred_labels, 'pred_labels_name': pred_labels_name}
        for one_info in table_info:
            for k, v in one_info.items():
                data_dict[k] = data_dict.get(k, [])
                data_dict[k].append(v)
        output_dict.update(data_dict)
        return output_dict

    def get_pred_labels(self, out_prob, labels, top_k=-1):
        pred_labels = []
        pred_labels_name = []
        for k, row_labels in enumerate(labels):
            n_pred = len(row_labels) if top_k == -1 else top_k
            pred_row_labels = out_prob[k][1:].argsort(dim=0, descending=True)[:n_pred].cpu().numpy()  # out_prob[0]: blank header
            pred_row_labels = [elem+1 for elem in pred_row_labels]  # add idx to 1 (for out_prob[0])
            pred_labels.append(pred_row_labels)
            pred_labels_name.append([self.label[elem] for elem in pred_row_labels])
        return pred_labels, pred_labels_name

    def sample_labels(self, labels_1d):
        # sampled label
        # mask_bool = torch.cuda.FloatTensor(self.n_classes).uniform_() > 0.9
        mask_bool = torch.cuda.FloatTensor(self.n_classes).uniform_() > 0.8
        mask = torch.tensor(mask_bool, dtype=torch.int)
        mask[labels_1d] = 2
        # print("m:", mask.shape, mask.sum(), (mask==1).sum(), (mask==2).sum())

        # target label
        new_labels = mask[mask!=0]
        new_labels[new_labels==1] = 0
        new_labels[new_labels==2] = 1

        # old_new label map
        old_idx = (mask==2).nonzero()
        new_idx = (new_labels==1).nonzero()
        idx_map = {}
        for k, idx in enumerate(old_idx):
            idx_map[int(idx)] = int(new_idx[k])
        new_labels_1d = torch.autograd.Variable(labels_1d.clone())
        for k, idx in enumerate(labels_1d):
            new_labels_1d[k] = idx_map[int(idx)]
        return mask, new_labels_1d

    def validate_seed_rows(self, table_info):
        for one_info in table_info:
            if one_info['num_rows'] > self.n_seed_rows:
                raise ValueError('invalid num rows')

    def get_meta(self, table_info):
        nrows = [one_info['num_rows'] for one_info in table_info]
        ncols = [one_info['num_cols'] for one_info in table_info]
        tids = [one_info['id'] for one_info in table_info]
        return nrows, ncols, tids

    @overrides
    def forward(self, table_info: Dict[str, str],
                indexed_headers: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:  # TODO: modify bert cache
        # indexed_cells: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        # initialize
        self.bert_embedder.eval()
        # self.validate_seed_rows(table_info)
        bs, n_rows, n_cols = TableUtil.get_max_row_col(table_info)
        nrows, ncols, tids = self.get_meta(table_info)
        table_mask = TableUtil.get_table_mask(table_info, bs, n_rows, n_cols, self.device)

        # pred prob
        bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, None, table_info, bs, n_rows, n_cols, self.cache_usage, self.bert_embedder, self.cache_util, self.device)
        row_embs, col_embs, n_rows_cls, n_cols_cls = self.get_tabemb(bert_header, bert_cell, n_rows, n_cols, bs, table_mask, nrows, ncols)

        # cls embs
        # cell_embs = self.cat_cell_embs(row_embs, col_embs, self.n_seed_rows, self.device)
        # if self.n_seed_rows == 2:
        #     cell_embs = self.mask_r1_2_embs(cell_embs, table_info)
        # else:
        #     cell_embs = self.mask_r1_3_embs(cell_embs, table_info)
        cell_embs = (col_embs[:, 0, 1, :] + row_embs[:, 0, 1, :]) / 2.0

        # label/prob
        labels_1d, labels = self.get_labels(table_info)
        labels_1d = torch.LongTensor(labels_1d).to(device=self.device)
        mask, sampled_labels_1d = self.sample_labels(labels_1d)

        if not self.training:
            mask = torch.ones(mask.shape, dtype=torch.int)
            sampled_labels_1d = labels_1d
        # mask = torch.ones(mask.shape, dtype=torch.int)
        # sampled_labels_1d = labels_1d

        out_prob_1d, out_prob = self.pred_prob(cell_embs, labels, mask)

        # calc loss func
        # loss = self.loss_func(out_prob_1d[:, mask.nonzero().flatten()], sampled_labels_1d)
        loss = self.loss_func(out_prob_1d, sampled_labels_1d)

        # pred labels
        # pred_labels, pred_labels_name = self.get_pred_labels(out_prob, labels)
        # out_prob_cp = self.mod_out_prob(out_prob_1d, pred_labels)

        # if random.random() < 0.01:
        #     for k, one_info in enumerate(table_info):
        #         print()
        #         print(one_info['seed_col0'])
        #         print(one_info['label'])
        #         print(pred_labels_name[k])

        # self.metrics['accuracy'](out_prob_cp[:, mask.nonzero().flatten()], sampled_labels_1d)
        self.metrics['accuracy'](out_prob_1d, sampled_labels_1d)
        output_dict = {'loss': loss}

        if not self.training:
            pred_labels, pred_labels_name = self.get_pred_labels(out_prob, labels, top_k=500)
            output_dict = self.add_metadata(table_info, output_dict, pred_labels, pred_labels_name)
        return output_dict


# if os.getenv('ft_type') == 'h_r1_2':
#     cls_embs = self.get_h_r1_2(row_embs, col_embs, self.n_seed_rows, self.device)
#     cls_embs = self.mask_h_r1_2_embs(cls_embs, table_info)
# elif os.getenv('ft_type') == 'h_r1_5':
#     pass
# else:
#     cls_embs = self.get_r1_3(row_embs, col_embs, self.n_seed_rows, self.device)
#     cls_embs = self.mask_r1_3_embs(cls_embs, table_info)


#     @staticmethod
#     def get_h_r1_2(row_embs, col_embs, n_seed_rows, device):
#         ave_embs = (row_embs + col_embs) / 2.0
#         bs = ave_embs.shape[0]
#         n_cell_rows = row_embs.shape[1]-2
#
#         col0_cell_embs = ave_embs[:, 2:, 1, :]
#         col0_embs = ave_embs[:, 1, 1, :]
#         for i in range(min(2, n_seed_rows)):
#             if n_cell_rows <= i:
#                 zeros = torch.zeros((bs, 768), device=device)
#                 col0_embs = torch.cat([col0_embs, zeros], dim=1)
#             else:
#                 col0_embs = torch.cat([col0_embs, col0_cell_embs[:, i, :]], dim=1)
#         return col0_embs
#
#     @staticmethod
#     def get_h_r1_5(row_embs, col_embs, n_seed_rows, device, row_ff):
#         ave_embs = (row_embs + col_embs) / 2.0
#         bs = ave_embs.shape[0]
#         n_cell_rows = row_embs.shape[1]-2
#
#         col0_cell_embs = ave_embs[:, 2:, 1, :]
#         col0_embs = ave_embs[:, 1, 1, :]
#         for i in range(min(6, n_seed_rows)):
#             if n_cell_rows <= i:
#                 zeros = torch.zeros((bs, 768), device=device)
#                 col0_embs = torch.cat([col0_embs, zeros], dim=1)
#             else:
#                 col0_embs = torch.cat([col0_embs, col0_cell_embs[:, i, :]], dim=1)
#         cls_embs = row_ff(col0_embs)
#         return cls_embs
#

#     @staticmethod
#     def mask_h_r1_2_embs(cls_embs, table_info):
#         for k, one_info in enumerate(table_info):
#             if one_info['num_rows'] == 1:
#                 cls_embs[k, (768*2):] = 0
#         return cls_embs



