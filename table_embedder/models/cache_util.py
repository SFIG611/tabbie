
import os
import copy
import torch
import numpy as np
# from kyotocabinet import *
# from kcdb import KCDB


class CacheUtil:
    def __init__(self, cache_usage, cell_db_path):
        if cache_usage == "read":
            self.cell_db = DB()
            self.cell_db.open(cell_db_path + "#bnum=20M#msiz=8G", DB.OREADER)
            print(self.cell_db.count())
            self.mem_db = DB()
            self.mem_db.open("-#bnum=20M#capcnt=20M", DB.OWRITER | DB.OCREATE)
            print(self.mem_db.count())
            KCDB.copy_db(self.cell_db, self.mem_db)
            self.cell_db.close()
            self.cell_db = self.mem_db
        elif cache_usage == "write":
            if os.path.exists(cell_db_path):
                raise ValueError("db: {} already exist".format(cell_db_path))
            self.cell_f_db = DB()
            self.cell_f_db.open(cell_db_path + "#bnum=30M#msiz=8G", DB.OWRITER | DB.OCREATE)
            self.cell_db = DB()
            self.cell_db.open("-#bnum=30M#capcnt=30M", DB.OWRITER | DB.OCREATE)

    # @staticmethod
    # def copy_db(db_read, db_write):
    #     cur = db_read.cursor()
    #     cur.jump()
    #     cnt = 0
    #     while True:
    #         record = cur.get(True)
    #         if not record: break
    #         db_write.set(record[0], "")  # create empty record for merge
    #         if cnt % 1000 == 0:
    #             print('dump done: {}, {}'.format(cnt, db_read.count()))
    #         cnt+=1
    #     cur.disable()
    #     db_write.merge([db_read], DB.MSET)
    #     db_write.close()

    def close(self):
        print('to file db')
        KCDB.copy_db(self.cell_db, self.cell_f_db)

    def post_process(self):
        if hasattr(self, 'cell_db'):
            self.cell_db.close()

    def save_header(self, table_info, bert_header):
        for k in range(len(table_info)):
            for j in range(len(table_info[k]["header"])):
                col2 = table_info[k]["header"][j]
                if self.cell_db.check(col2.lower()) == -1:
                    tmp = torch.autograd.Variable(bert_header[k, j].clone(), requires_grad=False).cpu().numpy()
                    self.cell_db.set(col2.lower(), tmp.tobytes())

    def save_cell(self, table_info, bert_cells):
        for k, one_info in enumerate(table_info):
            for i, row in enumerate(one_info['table_data_raw']):
                for j, cell in enumerate(row):
                    cell2 = table_info[k]['table_data_raw'][i][j]
                    if self.cell_db.check(cell2.lower()) == -1:
                        tmp = torch.autograd.Variable(bert_cells[k, i, j].clone(), requires_grad=False).cpu().numpy()
                        self.cell_db.set(cell2.lower(), tmp.tobytes())

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

    def load_header(self, table_info, bs, n_cols, device, f_dim=768):
        bert_header = np.zeros((bs, n_cols, f_dim), dtype=np.float32)
        for k, one_info in enumerate(table_info):
            headers = copy.deepcopy(one_info['header'])
            if 'replace_headers' in one_info and one_info['replace_headers'] is not None:
                for elem in one_info['replace_headers']:
                    headers[elem[0]] = elem[1]
            for j, header in enumerate(headers):
                if self.cell_db.check(header.lower()) != -1:
                    elem = self.cell_db.get(header.lower())
                    bert_header[k, j, :] = np.frombuffer(elem, dtype=np.float32)
                else:
                    raise ValueError("key {} doesn't exist".format(header.lower()))
        bert_header = torch.from_numpy(bert_header).to(device=device)
        return bert_header

    def load_cell(self, table_info, bs, n_row, n_col, table_data, device, f_dim=768):
        bert_cell = np.zeros((bs, n_row, n_col, f_dim), dtype=np.float32)
        for k in range(bs):
            n_row_k = len(table_info[k]["table_data_raw"])
            n_col_k = len(table_info[k]["table_data_raw"][0])
            for i in range(n_row_k):
                for j in range(n_col_k):
                    cell = table_data[k][i][j]
                    if self.cell_db.check(cell.lower()) != -1:
                        elem = self.cell_db.get(cell.lower())
                        y = np.frombuffer(elem, dtype=np.float32)
                        bert_cell[k, i, j, :] = y
                    else:
                        raise ValueError("key {} doesn't exist".format(cell.lower()))
        bert_cell = torch.from_numpy(bert_cell).to(device=device)
        return bert_cell

    @staticmethod
    def load_memory_db(db_path, memory=40):
        db = DB()
        db.open(db_path + "#bnum=20M#msiz={}G".format(str(memory)), DB.OREADER)
        mem_db = DB()
        mem_db.open("-#bnum=20M#capcnt=20M", DB.OWRITER | DB.OCREATE)
        CacheUtil.copy_db(db, mem_db)
        db.close()
        return mem_db

    @staticmethod
    def load_table_emb(emb_path, n_emb=-1):
        emb_db = CacheUtil.load_memory_db(emb_path)
        print('load: {}, cnt: {}'.format(emb_path, emb_db.count()))
        n_emb = emb_db.count() if n_emb == -1 else min(n_emb, emb_db.count())
        table_embs = np.zeros((n_emb, 768*2), dtype=np.float32)

        cur = emb_db.cursor()
        cur.jump()
        cnt = 0
        table_ids = []
        while True:
            rec = cur.get_str(True)
            if not rec: break
            table_ids.append(rec[0])
            table_embs[cnt, :] = np.frombuffer(emb_db.get(rec[0]), dtype=np.float32)
            cnt += 1
            if cnt == n_emb:
                break
        cur.disable()
        emb_db.close()
        return table_ids, table_embs

    @staticmethod
    def save_db(db, out_path, memory=16):
        db_out = DB()
        db_out.open(str(out_path) + "#bnum=20M#msiz={}G".format(memory), DB.OWRITER | DB.OCREATE)
        CacheUtil.copy_db(db, db_out)
        db.close()
        db_out.close()

    @staticmethod
    def init_memory_db():
        db = DB()
        db.open("-#bnum=20M#capcnt=20M", DB.OWRITER | DB.OCREATE)
        return db


