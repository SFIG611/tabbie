from typing import Dict, Optional
from overrides import overrides

import os
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder, ConditionalRandomField
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.models.archival import load_archive

from table_embedder.models.lib.bert_token_embedder import PretrainedBertEmbedder
from table_embedder.models.embedder_util import TableUtil
from table_embedder.models.lib.stacked_self_attention import StackedSelfAttentionEncoder

from table_embedder.models.cache_util import CacheUtil
from scripts.sato import Sato
from scripts.to_npy import ToNpy


@Model.register("finetune_sato")
class TableEmbedder(Model):

    def __init__(self, vocab: Vocabulary,
                 bert_embbeder: PretrainedBertEmbedder,
                 feedforward: FeedForward,
                 top_feedforward: FeedForward,
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
        self.top_feedforward = top_feedforward
        self.bert_embedder = bert_embbeder
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
        self.loss = torch.nn.BCELoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_class = 78
        self.num_max_row_pos = 35
        self.num_max_col_pos = 25

        if os.getenv('model_path') is not None and os.getenv('learn_type') != 'pred':
            self.init_weight()

        self.cls_col = np.load(os.getenv("clscol_path"))
        self.cls_row = np.load(os.getenv("clsrow_path"))
        self.cache_usage = os.getenv("cache_usage")
        if self.cache_usage is not None:
            self.cache_util = CacheUtil(self.cache_usage, os.getenv("cell_db_path"))
        else:
            self.cache_util = None

        id_path, npy_path = os.getenv('cache_id'), os.getenv('cache_npy')
        self.cell_id, self.cell_feats = ToNpy.load_feats(id_path, npy_path)

        self.opt_level = 'O0'
        self.label = Sato.load_label(label_path=os.getenv('label_path'))
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
                table_mask = TableUtil.add_cls_mask(table_mask, bs, n_rows, n_cols, self.device, nrows, ncols)
                col_embs = TableUtil.get_col_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_col, self.opt_level)
                row_embs = TableUtil.get_row_embs(bert_data, bs, n_rows, n_cols, table_mask, transformer_row, self.opt_level)
            else:
                row_embs = TableUtil.get_row_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_row, self.opt_level)
                col_embs = TableUtil.get_col_embs(ave_embs, bs, n_rows, n_cols, table_mask, transformer_col, self.opt_level)
            ave_embs = (row_embs + col_embs) / 2.0
        return row_embs, col_embs, n_rows, n_cols

    def get_labels(self, table_info, n_cols):
        bs = len(table_info)
        label_masks = torch.zeros((bs, n_cols), device=self.device)
        labels = torch.zeros((bs, n_cols), device=self.device)
        table_ids = np.empty(bs, dtype="U200")
        label_names = [[] for i in range(bs)]
        for k, one_info in enumerate(table_info):
            table_ids[k] = one_info['id']
            for j, col_id in enumerate(one_info['col_idx']):
                label_masks[k][col_id] = 1
                labels[k][col_id] = one_info['label_idx'][j]
                label_names[k].append(self.label[one_info['label_idx'][j]])
        return label_masks, labels, label_names, table_ids

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.metrics["accuracy"].get_metric(reset=reset)
        return {'accuracy': accuracy}

    def pred_prob(self, cls_embs, bs, n_cols, label_masks, labels):
        out_prob = self.top_feedforward(cls_embs)

        # to 1d
        out_prob_1d = out_prob.reshape(bs*n_cols, -1)
        labels_1d, label_masks_1d = labels.reshape(bs*n_cols), label_masks.reshape(bs*n_cols)
        label_masks_1d = label_masks_1d.bool()

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(out_prob_1d[label_masks_1d], labels_1d[label_masks_1d].long())
        self.metrics['accuracy'](out_prob_1d[label_masks_1d], labels_1d[label_masks_1d].long())
        return out_prob_1d, out_prob, loss

    def get_pred_labels(self, out_prob, table_info):
        bs = len(table_info)
        pred_labels = [[] for i in range(bs)]
        pred_label_names = [[] for i in range(bs)]
        for k, one_info in enumerate(table_info):
            for col_id in one_info['col_idx']:
                label_idx = out_prob[k][col_id].argsort(dim=0, descending=True)[0].cpu().numpy()
                pred_labels[k].append(label_idx)
                pred_label_names[k].append(self.label[int(label_idx)])
        return pred_labels, pred_label_names

    def get_meta(self, table_info):
        nrows = [one_info['num_rows'] for one_info in table_info]
        ncols = [one_info['num_cols'] for one_info in table_info]
        tids = [one_info['id'] for one_info in table_info]
        return nrows, ncols, tids

    @overrides
    def forward(self, table_info: Dict[str, str],
                indexed_headers: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        if not hasattr(self, 'bert_embedder'):
            self.bert_embedder = PretrainedBertEmbedder(pretrained_model='bert-base-uncased', top_layer_only=True)
            self.bert_embedder.to(device=self.device)
        self.bert_embedder.eval()

        bs, n_rows, n_cols = TableUtil.get_max_row_col(table_info)
        nrows, ncols, tids = self.get_meta(table_info)
        table_mask = TableUtil.get_table_mask(table_info, bs, n_rows, n_cols, self.device)
        label_masks, labels, label_names, table_ids = self.get_labels(table_info, n_cols)

        # to col_emb
        # bert_header, bert_cell = TableUtil.get_bert_emb(indexed_headers, indexed_cells, table_info, bs, n_rows, n_cols, self.cache_usage, self.bert_embedder, self.cache_util, self.device, self.cell_id, self.feats)
        bert_header, bert_cell = TableUtil.to_bert_emb(table_info, bs, n_rows, n_cols, self.device, self.cell_id, self.cell_feats)
        row_embs, col_embs, n_rows_cls, n_cols_cls = self.get_tabemb(bert_header, bert_cell, n_rows, n_cols, bs, table_mask, nrows, ncols)
        cls_embs = torch.cat([row_embs[:, 0, 1:, :], col_embs[:, 0, 1:, :]], dim=2)

        # pred prob
        _, out_prob, loss = self.pred_prob(cls_embs, bs, n_cols, label_masks, labels)
        pred_labels, pred_label_names = self.get_pred_labels(out_prob, table_info)

        # post-process
        output_dict = {'pred_label': pred_labels, 'loss': loss, 'table_id': table_ids, 'pred_label_names': pred_label_names}
        if not self.training:
            output_dict = self.debug(output_dict, bs, table_info, out_prob)
        return output_dict

    def debug(self, output_dict, bs, table_info, out_prob):
        labels_meta = [[] for i in range(bs)]
        labels_name = [[] for i in range(bs)]
        col_idx = [[] for i in range(bs)]
        out_prob_ = [[] for i in range(bs)]
        for k in range(bs):
            labels_meta[k] = table_info[k]['label_idx']
            col_idx[k] = table_info[k]['col_idx']
            for i, j in enumerate(col_idx[k]):
                labels_name[k].append(self.label[labels_meta[k][i]])
                if out_prob is not None:
                    out_prob_[k].append(out_prob[k][j])
        output_dict['label_idx'] = labels_meta
        if out_prob is not None:
            output_dict['out_prob'] = out_prob_
        output_dict['col_idx'] = col_idx
        output_dict['label_name'] = labels_name
        return output_dict



