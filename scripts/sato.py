
import copy
import json
import heapq
import random
import numpy as np
import pandas as pd

import gzip
import chardet

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from pathlib import Path
from scripts.util import Util


class SatoDev:
    @staticmethod
    def label_combs(base_dir):
        lines = Util.load_lines(base_dir/'s_test_0731.jsonl')
        label_cnt = {}
        label = Sato.load_label(base_dir/'label.csv')
        for line in lines:
            labels = '-'.join([label[label_idx] for label_idx in line['label_idx']])
            label_cnt[labels] = label_cnt.get(labels, [])
            label_cnt[labels].append(line['s3Link']+line['id'])
        for k, v in sorted(label_cnt.items(), key=lambda x: x[1], reverse=True):
            print(k)

        n_comb = 10
        valid_ids = []
        for label_comb, labels in label_cnt.items():
            if n_comb < len(labels):
                for table_id in random.sample(labels, n_comb):
                    valid_ids.append(table_id)

        new_lines = []
        for line in lines:
            if line['s3Link']+line['id'] in valid_ids:
                new_lines.append(line)

        Util.dump_lines(new_lines, base_dir/'s_test_small2.jsonl')
        exit()

        return label_cnt

    @staticmethod
    def _orig_label_cnt(csv_path, out_path):
        data = pd.read_csv(csv_path)
        label_cnt = {}
        for label_row in data['field_names'].tolist():
            for label in json.loads(label_row):
                label_cnt[label] = label_cnt.get(label, 0) + 1
        label_cnt = Util.dump_val_cnt(label_cnt, out_path)
        label_cnt.reset_index(drop=True, inplace=True)
        return label_cnt

    @staticmethod
    def _label_cnt(jsonl_path, out_path):
        test = Util.load_lines(jsonl_path)
        label_cnt = {}
        for line in test:
            for label in line['label_idx']:
                label_cnt[label] = label_cnt.get(label, 0) + 1
        label_cnt = Util.dump_val_cnt(label_cnt, out_path)
        label_cnt.reset_index(drop=True, inplace=True)
        return label_cnt

    @staticmethod
    def debug_test_label(base_dir):
        # label_cnt_orig = SatoDev._orig_label_cnt(base_dir/'single_test.csv', base_dir/'label_cnt_orig.csv')
        label_cnt = SatoDev._label_cnt(base_dir/'s_test_0731.jsonl', base_dir/'label_cnt.csv')
        # for col in label_cnt.columns:
        #     print(col, label_cnt[col].tolist() == label_cnt_orig[col].tolist())

    @staticmethod
    def _validate_lines(lines, validate_max_col=True):
        for line in lines:
            col_idx = line['col_idx']
            if len(col_idx) != len(line['label_idx']):
                raise ValueError('len(col_idx) != len(label_idx)')
            if not all(col_idx[i-1] <= col_idx[i] for i in range(1, len(col_idx))):
                raise ValueError('col_idx is not sorted')
            if validate_max_col and 20 < max(col_idx):
                raise ValueError('max(col_idx) > 20')

    @staticmethod
    def validate_data(base_dir):
        for fpath in ['s_train_0731.jsonl', 's_test_0731.jsonl']:
            lines = Util.load_lines(base_dir/fpath)
            SatoDev._validate_lines(lines)

    @staticmethod
    def _check_diff_lines(line1, line2, label):
        for key, val in line1.items():
            if val != line2[key] and key not in ['col_idx', 'table_data']:
                raise ValueError('unexpected different values')
        print(line1['col_idx'], line2['col_idx'])
        print(line1['table_data'])
        print(line2['table_data'], '\n')

    @staticmethod
    def debug_lines(base_dir, label_path):
        label = Sato.load_label(label_path)
        lines1 = Util.load_lines(base_dir/'s_test_0731.jsonl')
        # lines2 = Util.load_lines(base_dir/'s_test_swap.jsonl')
        # rows = []
        # for k, line1 in enumerate(lines1):
        #     lines2[k].pop('dtypes')
        #     if line1 != lines2[k]:
        #         rows.append(k)
        #         print('row: {}'.format(k))
        #         SatoDev._check_diff_lines(line1, lines2[k], label)
        # print('num diff rows: {}'.format(len(rows)))


class SatoEval:
    @staticmethod
    def evaluate(pred_path):
        labels = []
        preds = []
        for line in Util.load_lines(pred_path):
            labels += line['label_idx']
            preds += line['pred_label']
            if len(line['label_idx']) != len(line['pred_label']):
                raise ValueError('invalid')
        print(len(labels))
        f1_weighted = f1_score(labels, preds, average='weighted')
        f1_macro = f1_score(labels, preds, average='macro')
        return f1_weighted, f1_macro


class Sato:
    @staticmethod
    def _swap_label_col(line, max_n_cols=20):
        label_ind = [cid for cid in line['col_idx'] if cid >= max_n_cols]
        swap_ind = heapq.nlargest(len(label_ind), [cid for cid in range(max_n_cols) if cid not in line['col_idx']])[::-1]

        if len(label_ind) != 0:
            # update table data
            print('swap cols {}, {}'.format(label_ind, swap_ind))
            table = np.array(line['table_data'])
            for k, label_idx in enumerate(label_ind):
                swap_idx = swap_ind[k]
                tmp = copy.deepcopy(table[:, swap_idx])
                table[:, swap_idx] = copy.deepcopy(table[:, label_idx])
                table[:, label_idx] = tmp
            line['table_data'] = table.tolist()

            # update col_idx
            swap_map = dict(zip(label_ind, swap_ind))
            for k, label_idx in enumerate(line['col_idx']):
                if label_idx in swap_map.keys():
                    line['col_idx'][k] = swap_map[label_idx]
        return line

    @staticmethod
    def load_label(label_path):
        label = pd.read_csv(label_path)
        label_map = dict(zip(label['idx'], label['type']))
        return label_map

    @staticmethod
    def swap_label_cols(base_dir):
        for prefix in ['train', 'test']:
            lines = Util.load_lines(base_dir/'s_{}_0731_coltype.jsonl'.format(prefix))
            lines = [Sato._swap_label_col(line) for line in lines]
            Util.dump_lines(lines, base_dir/'s_{}_0731.jsonl'.format(prefix))

    @staticmethod
    def add_coltypes(base_dir):
        Util.add_coltype(base_dir/'s_train_0731_orig.jsonl', base_dir/'s_train_0731_coltype.jsonl')
        Util.add_coltype(base_dir/'s_test_0731_orig.jsonl', base_dir/'s_test_0731_coltype.jsonl')
        # Util.add_coltype(base_dir/'m_train.jsonl', base_dir/'m_train_types.jsonl')
        # Util.add_coltype(base_dir/'m_test.jsonl', base_dir/'m_test_types.jsonl')

    @staticmethod
    def _webtables_iter(path):
        # generate the next line of json(table)
        lines = []
        with gzip.open(path, 'rb') as f_in:
            iter_count = 0  # only count the # of succesfully yield dataframes
            for line_count, dataset in enumerate(f_in):
                try:
                    data = json.loads(dataset.decode('utf-8'))
                    lines.append(data)
                    # yield (iter_count, data)
                    iter_count += 1
                except UnicodeDecodeError:
                    encoding = chardet.detect(dataset)['encoding']
                    try:
                        data = json.loads(dataset.decode(encoding))
                        lines.append(data)
                        # yield (iter_count, data)
                        iter_count += 1
                    except Exception as e:
                        # print('Cannot parse:', e)
                        continue
                    continue
        return lines

    @staticmethod
    def get_valid_types(TYPENAME):
        import os
        with open(os.path.join(os.environ['BASEPATH'], 'configs', 'types.json'), 'r') as typefile:
            valid_types = json.load(typefile)[TYPENAME]
        return valid_types

    @staticmethod
    def extract_webtables(data, locator, dataset_id=None, exact_num_fields=None, min_fields=None, max_fields=None,
                          valid_fields=None, line_no=0):
        # if dataset_id is set, only extract if there's a match
        from collections import OrderedDict
        try:
            # webtables are not uniquely identified by pageTitle + tableNum,
            # TO distinguish between tables, add a index with respect to location in the conatining file.
            if data['hasHeader'] and (data['headerPosition'] == 'FIRST_ROW'):
                d_id = '{}-{}-{}'.format(line_no, data['pageTitle'], data['tableNum'])

                # table name not matching
                if dataset_id is not None and d_id != dataset_id:
                    return
                header_row_index = data.get('headerRowIndex', 0)
                data_as_dict = OrderedDict()
                for raw_cols in data['relation']:
                    header_row = raw_cols[header_row_index]
                    raw_cols.pop(header_row_index)

                    parsed_values = pd.Series([None if (v == '-') else v for v in raw_cols])
                    try:
                        parsed_values = pd.to_numeric(parsed_values, errors='raise')
                    except:
                        # print('CAN"T PARASE')
                        pass
                    # parsed_values = parsed_values.replace(value='-', None)
                    data_as_dict[header_row] = parsed_values

                df = pd.DataFrame(data_as_dict)

                num_fields = len(df.columns)

                if exact_num_fields:
                    if num_fields != exact_num_fields: return
                if min_fields:
                    if num_fields < min_fields: return
                if max_fields:
                    if num_fields > max_fields: return

                # If specified, only return the valid fields
                if valid_fields is not None:
                    df = df.iloc[:, valid_fields]

                if df is not None:
                    header, cells = df.columns.tolist(), df.to_numpy().tolist()
                    table = np.vstack([header, cells]).tolist()
                    return {
                        'table_data': table,
                        'id': d_id,
                        'locator': locator
                    }

                result = {
                    'df': df,
                    'dataset_id': d_id,
                    'locator': locator
                }
                return result
            else:
                return
        except Exception as e:
            print("Exception in table extraction: ", e)
            return

    @staticmethod
    def decompress_gzip(viznet_dir, out_dir):
        for fpath in viznet_dir.glob('**/*.json.gz'):
            key = fpath.parent.parent.name + '/warc/' + fpath.stem
            (out_dir/key).parent.mkdir(exist_ok=True, parents=True)
            lines = Sato._webtables_iter(fpath)  # read valid rows
            Util.dump_lines(lines, out_dir/key)

    @staticmethod
    def to_meta_dict(meta_csv):
        meta = pd.read_csv(meta_csv)
        meta_dict = {}
        for idx, row in meta.iterrows():
            fid, title_idx = row['table_id'].split('.gz+')[0], row['table_id'].split('.gz+')[1]
            meta_dict[fid] = meta_dict.get(fid, {})
            meta_dict[fid][title_idx] = meta_dict[fid].get(title_idx, {})
            meta_dict[fid][title_idx]['col_idx'] = json.loads(row['field_list'])
            meta_dict[fid][title_idx]['label_idx'] = json.loads(row['field_names'])
            meta_dict[fid][title_idx]['row_id'] = int(Path(row['table_id'].split('.gz+')[1].split('-')[0]).stem)
        return meta_dict

    @staticmethod
    def dump_dataset_old(meta_csv, out_path, json_data_dir):
        meta_dict = Sato.to_meta_dict(meta_csv)
        fout = open(str(out_path), 'w')
        for fname, title_dict in meta_dict.items():
            lines = Util.load_lines(json_data_dir/fname)
            for k, line in enumerate(lines):
                key = '-'.join([str(k), line['pageTitle'], str(line['tableNum'])])
                if key not in title_dict.keys():
                    continue
                line['col_idx'], line['label_idx'] = title_dict[key]['col_idx'], title_dict[key]['label_idx']
                line['id'], line['row_id'] = key, title_dict[key]['row_id']
                line['table_data'] = np.array(line['relation']).transpose()[1:, :].tolist()
                fout.write(json.dumps(line)+'\n')
        fout.close()

    @staticmethod
    def dump_cv(base_dir):
        for fname in ['webtables1-p1_type78.json', 'webtables2-p1_type78.json']:
            fpath = base_dir / 'train_test_split/{}'.format(fname)
            data = json.load(open(fpath))
            # all = np.array(Util.shuf(data['train']+data['test']))
            all = np.array(data['train']+data['test'])
            kf = KFold(n_splits=5)
            for k, (train_idx, test_idx) in enumerate(kf.split(all)):
                one_row = {'train': all[train_idx].tolist(), 'test': all[test_idx].tolist()}
                out_dir = base_dir / 'train_test_split{}'.format(k)
                out_dir.mkdir(exist_ok=True, parents=True)
                with open(out_dir/'{}'.format(fname), 'w') as fout:
                    json.dump(one_row, fout)

    @staticmethod
    def _add_label_names(lines, label):
        for k, line in enumerate(lines):
            lines[k]['label_names'] = [label[elem] for elem in line['label_idx']]
        return lines

    @staticmethod
    def add_label_names(fpath, out_path, label_path):
        label = Sato.load_label(label_path)
        lines = Util.load_lines(fpath)
        lines = Sato._add_label_names(lines, label)
        Util.dump_lines(lines, out_path)

    @staticmethod
    def add_label_header(test_path, out_path, label):
        lines = Util.load_lines(test_path)
        for k, line in enumerate(lines):
            lines[k]['header'] = ['' for elem in range(len(lines[k]['table_data'][0]))]
            for j, label_idx in enumerate(lines[k]['label_idx']):
                lines[k]['header'][line['col_idx'][j]] = label[label_idx]
            lines[k]['table_data'] = np.vstack([line['header'], line['table_data']]).tolist()
        Util.dump_lines(lines, out_path)

    @staticmethod
    def filter_tables(webtables, table_ids):
        tables = []
        for table in webtables:
            key = '{}+{}'.format(table['locator'], table['id'])
            if key not in table_ids:
                continue
            tables.append(table)
        return tables

    @staticmethod
    def load_tables(data_dir):
        new_lines = []
        for fpath in data_dir.glob('**/*.json'):
            locator = fpath.parent.parent.name + '/warc/' + fpath.stem + '.json.gz'
            lines = Util.load_lines(fpath)
            for k, line in enumerate(lines):
                line = Sato.extract_webtables(line, locator, line_no=k)
                if line is None:
                    continue
                if 'table_data' not in line:
                    raise ValueError('error: no table_data key')
                new_lines.append(line)
        return new_lines

    @staticmethod
    def to_new_tables(lines_small, lines_raw_dict):
        cnt = 0
        for k, line in enumerate(lines_small):
            locator = Path(line['s3Link']).parent.parent.name + '/warc/' + Path(line['s3Link']).stem.rstrip('.warc') + '.json.gz'
            idx = '{}+{}'.format(locator, line['id'])
            if idx not in lines_raw_dict:
                continue
            table_old = np.array(copy.deepcopy(line['table_data']))
            table_new = np.array(lines_raw_dict[idx]['table_data']).astype(str)
            table_new[table_new=='nan'] = ''
            lines_small[k]['table_data'] = np.array(table_new)[1:, :].tolist()
        return lines_small

    @staticmethod
    def to_dict(lines_raw):
        lines_dict = {}
        for k, line in enumerate(lines_raw):
            lines_dict['{}+{}'.format(line['locator'], line['id'])] = line
        return lines_dict

    @staticmethod
    def create_new_dataset(data_dir, dump_all=False):
        if dump_all:
            tables1 = Sato.load_tables(data_dir/'input_json/1438042986022.41/warc')
            tables2 = Sato.load_tables(data_dir/'input_json/1438042987171.38/warc')
            tables = tables1 + tables2
            Util.dump_lines(tables, data_dir/'tables.jsonl')
        tables = Util.load_lines(data_dir/'tables.jsonl')

        # dump train/test
        for dname in ['train', 'test']:
            meta = pd.read_csv(data_dir/'s_{}_meta.csv'.format(dname))
            lines_raw = Sato.filter_tables(tables, Util.to_dict(meta['table_id'].tolist()))
            Util.dump_lines(lines_raw, data_dir/'{}_raw.jsonl'.format(dname))

        # dump new data
        # TODO: use meta
        # for dname in ['train', 'test']:
        #     lines_raw = Util.load_lines(data_dir / '{}_raw.jsonl'.format(dname))
        #     lines_small = Util.load_lines(data_dir/'s_{}_0731.jsonl'.format(dname))
        #     lines_small_new = Sato.to_new_tables(lines_small, Sato.to_dict(lines_raw))
        #     Util.dump_lines(lines_small_new, data_dir/'{}_0731.jsonl'.format(dname))

    @staticmethod
    def dump_shuf(data_dir):
        train_raw, test_raw = Util.load_lines(data_dir/'train_0731.jsonl'), Util.load_lines(data_dir/'test_0731.jsonl')
        train, test = Util.shuf(train_raw), Util.shuf(test_raw)
        train, test = train[:4000], test[:2000]
        Util.dump_lines(train, data_dir/'train4K.jsonl')
        Util.dump_lines(test, data_dir/'test2K.jsonl')

    @staticmethod
    def view_label(data_dir, label):
        lines = Util.load_lines(data_dir/'s_test_small.jsonl')
        label_cnt = {}
        for line in lines:
            for elem in line['label_idx']:
                label_cnt[label[elem]] = label_cnt.get(label[elem], 0) + 1
        for k, v in sorted(label_cnt.items(), key=lambda x:x[1], reverse=True):
            print(k, v)
        # print(label_cnt)
        exit()

    @staticmethod
    def test4K(data_dir):
        lines = Util.load_lines(data_dir/'test.jsonl')
        test_meta = pd.read_csv(data_dir/'s_test_meta.csv')
        for line in lines:
            key = '{}+{}'.format(line['locator'], line['id'])
            print(test_meta[test_meta['table_id'] == key])


def main():
    print('test')
    exit()
    # data_dir = Path('/home/hiida/code/sato_data/')
    data_dir = Path('/home/hiida/code/table_emb/exp/s_3880/0731')
    Sato.create_new_dataset(data_dir)
    # exit()
    # test = Sato.test4K(data_dir)
    # print(test)
    # exit()

    # Sato.create_new_dataset(data_dir)
    # Sato.add_label(data_dir)
    # label = Sato.load_label(data_dir/'label.csv')
    # Sato.view_label(data_dir, label)
    # exit()

    # data_dir = Path('/local/sato_orig')
    # Sato.dump_shuf(data_dir)
    # Sato.create_new_dataset(data_dir)
    # Sato.compare(data_dir)

    # # base_dir = Path('/mnt/nfs/work1/miyyer/hiida/data/sato/data/0731')
    # base_dir = Path('/local/sato_orig')
    # Sato.create_dataset(base_dir/'s_test_meta.csv', base_dir/'extract/out/headers/type78')

    # base_dir = Path('../exp/s_3880')
    # Sato.add_label_names(base_dir/'s_train_small.jsonl', base_dir/'s_train_small_dev.jsonl', base_dir/'label.csv')
    # Sato.add_label_names(base_dir/'s_test_small.jsonl', base_dir/'s_test_small_dev.jsonl', base_dir/'label.csv')
    # exit()

    # base_dir = Path('/home/hiida/code/sato_data')
    # label = Sato.load_label(base_dir/'label.csv')
    # Sato.add_label_header(base_dir/'test_small.jsonl', base_dir/'test_small_viz.jsonl', label)
    # lines = Util.load_lines(base_dir/'test_small_viz.jsonl')
    # Util.jsonl_to_html(lines, base_dir/'test_small_viz')

    # Sato.add_label_header(base_dir/'s_test_small.jsonl', base_dir/'s_test_small_viz.jsonl', label)
    # lines = Util.load_lines(base_dir/'s_test_small_viz.jsonl')
    # Util.jsonl_to_html(lines, base_dir/'s_test_small_viz')
    exit()
    # Sato.decompress_gzip(base_dir/'input_gzip', base_dir/'input_json')
    # Sato.dump_dataset_old(base_dir/'s_train_meta.csv', base_dir/'s_train_0731.jsonl', base_dir/'input_json')
    # Sato.dump_dataset_old(base_dir/'s_test_meta.csv', base_dir/'s_test_0731.jsonl', base_dir/'input_json')
    # Sato.add_coltypes(base_dir)
    # Sato.swap_label_cols(base_dir)
    exit()

    label_combs = SatoDev.label_combs(base_dir)
    for label in label_combs:
        print(label)
    exit()
    # exit()

    SatoDev.debug_test_label(base_dir)
    SatoDev.validate_data(base_dir)
    SatoDev.debug_lines(base_dir, base_dir/'label.csv')

    base_dir = Path('/home/ubuntu/table_embedding/Sato/extract/out')
    Sato.dump_cv(base_dir)


if __name__ == '__main__':
    main()



