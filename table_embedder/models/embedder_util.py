
import os
import json
import glob
import copy
import torch
import numpy as np
from pathlib import Path
# from cache_util import CacheUtil
# from TaBERT.table_bert import Table, Column


class TableUtil:
    @staticmethod
    def to_crf_fmt(max_len, out_prob, labels, col_ind, n_class, device):
        bs = len(out_prob)
        out_prob_crf = torch.zeros((bs, max_len, n_class), device=device)
        labels_crf = torch.zeros((bs, max_len), device=device)
        labels_mask_crf = torch.zeros((bs, max_len), device=device)
        for k, one_table_col in enumerate(col_ind):
            for j, one_col in enumerate(one_table_col):
                out_prob_crf[k][j] = out_prob[k][one_col]
                labels_crf[k][j] = labels[k][one_col]
                labels_mask_crf[k][j] = 1
        return out_prob_crf, labels_crf, labels_mask_crf

    @staticmethod
    def get_col_ind(table_info):
        col_ind = []
        max_len = -1
        for one_info in table_info:
            col_ind.append(one_info['col_idx'])
            if max_len < len(one_info['col_idx']):
                max_len = len(one_info['col_idx'])
        return col_ind, max_len

    # @staticmethod
    # def add_cls_mask(table_mask, table_info, bs, n_rows, n_cols, device):
    #     cls_mask_col2 = torch.ones([bs, 1, n_cols-1], device=device)
    #     cls_mask_row2 = torch.ones([bs, n_rows+1, 1], device=device)
    #     for k, one_info in enumerate(table_info):
    #         cls_mask_col2[k, 0, one_info['num_cols']:] = 0
    #         cls_mask_row2[k, (one_info['num_rows'] + 2):, 0] = 0
    #     table_mask = torch.cat([cls_mask_col2, table_mask], dim=1)
    #     table_mask = torch.cat([cls_mask_row2, table_mask], dim=2)
    #     return table_mask

    @staticmethod
    def add_cls_mask(table_mask, bs, max_rows_cls, max_cols_cls, device, nrows, ncols):
        cls_mask_col = torch.ones([bs, 1, max_cols_cls-1], device=device)
        cls_mask_row = torch.ones([bs, max_rows_cls+1, 1], device=device)
        for k in range(len(nrows)):
            cls_mask_col[k, 0, ncols[k]:] = 0
            cls_mask_row[k, (nrows[k]+2):, 0] = 0
        table_mask = torch.cat([cls_mask_col, table_mask], dim=1)
        table_mask = torch.cat([cls_mask_row, table_mask], dim=2)
        return table_mask

    @staticmethod
    def add_cls_tokens(bert_header, bert_data, cls_row, cls_col, bs, n_rows_cls, n_cols_cls):
        cls_col = cls_col.expand((bs, 1, n_cols_cls-1, 768))
        # print(cls_col.shape, cls_col.dtype)
        # print(bs, n_cols, bert_header.shape, bert_data.shape, bert_header.dtype, bert_data.dtype)
        bert_data = torch.cat([cls_col, bert_header.reshape(bs, 1, n_cols_cls-1, 768), bert_data], dim=1)
        cls_row = cls_row.expand((bs, n_rows_cls+1, 1, 768))
        bert_data = torch.cat([cls_row, bert_data], dim=2)
        return bert_data

    @staticmethod
    def get_row_embs(bert_data, bs, max_rows, max_cols, table_mask, transformer_row, opt_level, f_dim=768):
        bert_data_mod = bert_data * torch.unsqueeze(table_mask, 3).expand_as(bert_data).float()
        table_mask_row = table_mask.reshape(bs*(max_rows+1), max_cols)
        bert_data_mod = bert_data_mod.reshape(bs*(max_rows+1), max_cols, f_dim)
        if opt_level in ['O1', 'O2', 'O3']:
            row_embs = transformer_row(bert_data_mod.half(), table_mask_row)  # (10, 4, 768)
        else:
            row_embs = transformer_row(bert_data_mod, table_mask_row)  # (10, 4, 768)
        row_embs = row_embs.reshape(bs, max_rows+1, max_cols, f_dim)
        return row_embs

    @staticmethod
    def get_col_embs(bert_data, bs, max_rows, max_cols, table_mask, transformer_col, opt_level, f_dim=768):
        bert_data_mod = bert_data * torch.unsqueeze(table_mask, 3).expand_as(bert_data).float()
        bert_data_mod = bert_data_mod.permute(0, 2, 1, 3)
        table_mask_col = table_mask.permute(0, 2, 1)
        bert_data_mod = bert_data_mod.reshape(bs*max_cols, (max_rows+1), f_dim)  # (bs*n_cols, n_rows+1, f_dim)
        table_mask_col = table_mask_col.reshape(bs*max_cols, (max_rows+1))
        if opt_level in ['O1', 'O2', 'O3']:
            col_embs = transformer_col(bert_data_mod.half(), table_mask_col)  # (bs*n_cols, n_rows+1, f_dim)
        else:
            col_embs = transformer_col(bert_data_mod, table_mask_col)  # (bs*n_cols, n_rows+1, f_dim)
        col_embs = col_embs.reshape(bs, max_cols, max_rows+1, f_dim)
        col_embs = col_embs.permute(0, 2, 1, 3)
        return col_embs

    @staticmethod
    def load_header_npy(table_info, bs, n_cols, cell_id, feats, device, f_dim=768):
        bert_header = np.zeros((bs, n_cols, f_dim), dtype=np.float32)
        for k, one_info in enumerate(table_info):
            headers = copy.deepcopy(one_info['header'])
            if 'replace_headers' in one_info and one_info['replace_headers'] is not None:
                for elem in one_info['replace_headers']:
                    headers[elem[0]] = elem[1]
            for j, header in enumerate(headers):
                bert_header[k, j, :] = feats[cell_id[header.lower()]].astype(np.float32)
        bert_header = torch.from_numpy(bert_header).to(device=device)
        return bert_header

    @staticmethod
    def load_cell_npy(table_info, bs, n_row, n_col, table_data, cell_id, feats, device, f_dim=768):
        bert_cell = np.zeros((bs, n_row, n_col, f_dim), dtype=np.float32)
        for k in range(bs):
            n_row_k = len(table_info[k]["table_data_raw"])
            n_col_k = len(table_info[k]["table_data_raw"][0])
            for i in range(n_row_k):
                for j in range(n_col_k):
                    cell = table_data[k][i][j]
                    bert_cell[k, i, j, :] = feats[cell_id[cell.lower()]].astype(np.float32)
        bert_cell = torch.from_numpy(bert_cell).to(device=device)
        return bert_cell

    @staticmethod
    def to_bert_emb(table_info, bs, n_rows, n_cols, device, cell_id, cell_feats, f_dim=768):
        table_data_fake = TableUtil.get_table_data(table_info)
        bert_header = TableUtil.load_header_npy(table_info, bs, n_cols, cell_id, cell_feats, device=device, f_dim=f_dim)
        bert_cell = TableUtil.load_cell_npy(table_info, bs, n_rows, n_cols, table_data_fake, cell_id, cell_feats, device=device, f_dim=f_dim)
        return bert_header, bert_cell

    @staticmethod
    def get_bert_emb(indexed_headers, indexed_cells, table_info, bs, n_rows, n_cols, cache_usage, bert_embedder, cache_util, device, f_dim=768):
        if cache_usage is None:
            bert_header = bert_embedder(indexed_headers['bert']['token_ids'])[:, :, 0, :]  # (bs, n_cols, n_words, 768)
            bert_cell = bert_embedder(indexed_cells['bert']['token_ids'])[:, :, :, 0, :]  # (bs, n_rows, n_cols, n_words, 768)
        elif cache_usage == 'read':
            table_data_fake = TableUtil.get_table_data(table_info)
            bert_header = cache_util.load_header(table_info, bs, n_cols, device=device, f_dim=f_dim)
            bert_cell = cache_util.load_cell(table_info, bs, n_rows, n_cols, table_data_fake, device=device, f_dim=f_dim)
        elif cache_usage == "write":
            bert_header = bert_embedder(indexed_headers['bert']['token_ids'])[:, :, 0, :]  # (bs, n_cols, n_words, 768)
            bert_cell = bert_embedder(indexed_cells['bert']['token_ids'])[:, :, :, 0, :]  # (bs, n_rows, n_cols, n_words, 768)
            cache_util.save_header(table_info, bert_header)
            cache_util.save_cell(table_info, bert_cell)
        else:
            raise ValueError('invalid cache usage: {}'.format(cache_usage))
        return bert_header, bert_cell

    @staticmethod
    def get_max_row_col(table_info):
        bs = len(table_info)
        max_rows, max_cols = -1, -1
        for one_info in table_info:
            if max_rows < one_info['num_rows']:
                max_rows = one_info['num_rows']
            if max_cols < one_info['num_cols']:
                max_cols = one_info['num_cols']
        return bs, max_rows, max_cols

    @staticmethod
    def get_cell_labels(table_info, bs, max_rows, max_cols, device):
        labels = torch.zeros(bs, max_rows, max_cols, dtype=torch.long, device=device)
        for k, one_info in enumerate(table_info):
            if ('replace_cell_ids' not in one_info) or one_info['replace_cell_ids'] is None:
                continue
            for j, (row_id, col_id) in enumerate(one_info['replace_cell_ids']):
                if one_info['table_data_raw'][row_id][col_id] == one_info['replace_cell_words'][j]:
                    continue
                labels[k, row_id, col_id] = 1
        return labels

    @staticmethod
    def get_header_labels(table_info, bs, max_cols, device):
        labels = torch.zeros(bs, 1, max_cols, dtype=torch.long, device=device)
        for k, one_info in enumerate(table_info):
            print(one_info['replace_headers'])
            print(one_info['header'])
            if ('replace_headers' not in one_info) or one_info['replace_headers'] is None:
                continue
            for (col_id, val) in one_info['replace_headers']:
                if one_info['header'][col_id] == val:
                    continue
                print(k, col_id, 111111)
                labels[k, 0, col_id] = 1
        return labels

    @staticmethod
    def get_table_mask(table_info, bs, max_rows, max_cols, device):
        mask = torch.ones(bs, max_rows+1, max_cols, device=device)
        for k, one_info in enumerate(table_info):
            mask[k, (one_info['num_rows']+1):, :] = 0
            mask[k, :, one_info['num_cols']:] = 0
        return mask

    @staticmethod
    def get_table_mask_blank(table_info, table_mask):
        table_mask_blank = copy.deepcopy(table_mask)
        for k, one_info in enumerate(table_info):
            for elem in one_info['blank_loc']:
                table_mask_blank[k, elem[0], elem[1]] = 0
        return table_mask_blank

    @staticmethod
    def get_table_data(table_info):
        table_data_fake = [copy.deepcopy(table_info[i]["table_data_raw"]) for i in range(len(table_info))]
        for k, one_info in enumerate(table_info):
            if ('replace_cell_ids' not in one_info) or (one_info["replace_cell_ids"] is None):
                continue
            for j, (row_id, col_id) in enumerate(one_info["replace_cell_ids"]):
                if col_id > 19:
                    raise ValueError("something wrong")
                table_data_fake[k][row_id][col_id] = one_info["replace_cell_words"][j]
        return table_data_fake

    @staticmethod
    def get_cells(table_info, n_rows, n_cols):
        cells = np.empty((len(table_info), n_rows, n_cols), dtype="U30")
        for k, one_info in enumerate(table_info):
            for i in range(n_rows):
                for j in range(n_cols):
                    try:
                        cells[k][i][j] = one_info['table_data_raw'][i][j]
                    except:
                        cells[k][i][j] = ''
        return cells

    @staticmethod
    def get_replace_cells(table_info, cells):
        replaced_cells = copy.deepcopy(cells)
        for k, one_info in enumerate(table_info):
            if one_info['replace_cell_ids'] is None:
                continue
            for l in range(0, len(one_info['replace_cell_ids'])):
                i = one_info['replace_cell_ids'][l][0]
                j = one_info['replace_cell_ids'][l][1]
                fake_cell = one_info['replace_cell_words'][l]
                replaced_cells[k][i][j] = fake_cell
        return replaced_cells

    @staticmethod
    def load_table(input_jsonl):
        table_data = {}
        print('load: {}'.format(input_jsonl))
        with open(input_jsonl, "r") as data_file:
            for i, line in enumerate(data_file):
                json_data = json.loads(line)
                table_id = json_data['id'].replace("/", "_").replace(".txt", "")
                table_data[table_id] = np.array(json_data['table_data']).tolist()
        return table_data


