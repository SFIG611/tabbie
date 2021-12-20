
import json
import numpy as np
from pathlib import Path
# from kyotocabinet import *


class ToNpy:
    @staticmethod
    def load_feats(id_path, npy_path):
        cell_id = ToNpy.load_cell_json(id_path)
        feats = np.load(npy_path)
        return cell_id, feats

    @staticmethod
    def load_cell_json(json_path):
        with open(json_path) as f:
            cell_json = json.loads(f.read())
        return cell_json

    @staticmethod
    def copy_db(input_db, out_db):
        cur = input_db.cursor()
        cur.jump()
        cnt = 0
        while True:
            record = cur.get(True)
            if not record: break
            out_db.set(record[0], "")  # create empty record for merge
            cnt+=1
            if cnt % 100000 == 0:
                print('dump done: {}, {}'.format(cnt, input_db.count()))
        cur.disable()
        out_db.merge([input_db], DB.MSET)
        return out_db

    @staticmethod
    def load_cid(cell_feats_path, cell_id_path):
        # load cell feats
        print('load: {}'.format(cell_feats_path))
        cell_feats = np.load(cell_feats_path)
        print('load done: {}'.format(cell_feats_path))

        # load cell id
        with open(cell_id_path) as f:
            cell_id = f.readlines()
            for i, elem in enumerate(cell_id):
                cell_id[i] = cell_id[i].replace('\n', '')
        print('load cid: {}'.format(cell_id_path))

        # to cid_map
        cid_map = {}
        for i, cell in enumerate(cell_id):
            cid_map[cell] = i
        return cell_feats, cid_map

    @staticmethod
    def get_records(db):
        print(db.count())
        cur = db.cursor()
        cur.jump()
        records = []
        while True:
            record = cur.get(True)
            if not record: break
            records.append(record)
            # print(record[0].decode('utf-8'))
        return records

    @staticmethod
    def load_mem_db(db_path, msiz='150G', capcnt='200M'):
        db, mem_db = DB(), DB()
        db.open("{}#bnum=100M#msiz={}".format(db_path, msiz), DB.OREADER)
        mem_db.open("-#bnum=100M#capcnt={}".format(capcnt), DB.OWRITER | DB.OCREATE)
        ToNpy.copy_db(db, mem_db)
        db.close()
        return mem_db

    @staticmethod
    def dump_cell_feats(kch_path, out_dir):
        out_dir.mkdir(exist_ok=True, parents=True)
        cell_db = ToNpy.load_mem_db(kch_path, msiz='350G')
        cid_map = {}
        cell_feats = np.zeros((cell_db.count(), 768), dtype=np.float16)
        for k, rec in enumerate(ToNpy.get_records(cell_db)):
            cid_map[rec[0].decode('utf-8')] = k
            cell_feats[k] = np.frombuffer(rec[1], dtype=np.float32).astype(np.float16)
        with open(out_dir/'cid_map.json', 'w') as fout:  # TODO: delete
            fout.write(json.dumps(cid_map))
        np.save(out_dir/'cell_feats.npy', cell_feats)

    @staticmethod
    def json_to_txt(json_path, out_path):
        cell_json = ToNpy.load_cell_json(json_path)
        with open(out_path, 'w') as fout:
            for w, idx in sorted(cell_json.items(), key=lambda x: x[1]):
                fout.write(w+'\n')


class TestNpy:
    @staticmethod
    def check_cache(cache_dir):
        # cell_id, feats = ToNpy.load_feats(cache_dir/'cid_map.json', cache_dir/'cell_feats.npy')
        cell_feats, cid_map = ToNpy.load_cid(cache_dir / 'cell_feats.npy', cache_dir / 'cell_id.txt')
        print(cell_feats.shape, len(cid_map))
        # print(len(cell_id))
        # print(feats.shape)
        # if len(cell_id) != feats.shape[0]:
        #     raise ValueError('err')

    @staticmethod
    def compare_cache(cache_dir1, cache_dir2):
        cell_feats, cid_map = ToNpy.load_cid(cache_dir1 / 'cell_feats.npy', cache_dir1 / 'cell_id.txt')
        cell_feats2, cid_map2 = ToNpy.load_cid(cache_dir2 / 'cell_feats.npy', cache_dir2 / 'cell_id.txt')
        cell_feats2 = cell_feats2.astype(np.float16)
        print(cell_feats.dtype, cell_feats2.dtype)
        print(cell_feats.shape, cell_feats2.shape, len(cid_map), len(cid_map2))
        for cell, idx in cid_map.items():
            idx2 = cid_map2[cell]
            # print(cell_feats[idx][:3])
            # print(cell_feats2[idx2][:3])
            ret = np.allclose(cell_feats[idx], cell_feats2[idx2], atol=5e-03)
            print(ret)
        exit()


def main():
    # base_dir = Path('/home/hiida/tabbie_ex/data/sato')
    # ToNpy.dump_cell_feats(base_dir/'sato_cache.kch', base_dir)

    # base_dir = Path('/home/hiida/tabbie_v0/data/ft_cell/log_cache0')
    # ToNpy.dump_cell_feats(base_dir/'cell_db_part0.kch', base_dir)
    # ToNpy.json_to_txt(base_dir/'cid_map.json', base_dir/'cell_id.txt')

    cache_dir = Path('/home/hiida/tabbie_v0/data/ft_cell/log_cache0')
    cache_dir2 = Path('/home/hiida/tabbie_v0/data/ft_cell')
    # TestNpy.check_cache(cache_dir)
    TestNpy.compare_cache(cache_dir, cache_dir2)
    # TestNpy.check_cache(base_dir/'cell_feats.npy', base_dir/'cid_map.json')

    print('test')
    print('test2')


if __name__ == '__main__':
    main()


