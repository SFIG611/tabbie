import os
import re
import json
import yaml
import copy
import random
import tarfile
import subprocess
from shutil import copyfile
# from numpy.random import default_rng

from pathlib import Path
import numpy as np
import pandas as pd


class Util:
    @staticmethod
    def load_yaml(yaml_path, encoding="utf-8-sig"):
        with open(yaml_path, encoding=encoding) as f:
            return yaml.safe_load(f.read())

    @staticmethod
    def make_tarfile(out_fname, input_dir):
        with tarfile.open(out_fname, "w:gz") as fout:
            fout.add(input_dir, arcname=os.path.basename(input_dir))
            print('dump {}'.format(Path(input_dir).parent.resolve()/Path(out_fname).name))

    @staticmethod
    def filter_span(input_str):
        input_str = re.sub(r'<span style.*?/span>', '', input_str)
        input_str = re.sub(r'<.*?>','', input_str)
        input_str = re.sub(r'(?P<url>https?://[^\s]+)', '', input_str)
        return input_str

    @staticmethod
    def remove_bracket(target_str, idx):
        match = re.match(r'\[(.*)\|(.*)\]', target_str)
        # re.match(r'.*<A>(.*?)<B>.*', target_str).group(1)  # shortest match
        if match is None:
            return target_str
        else:
            return match.group(idx)

    @staticmethod
    def dump_val_cnt(val_cnt_dict, out_path):
        val_cnt_df = pd.DataFrame(val_cnt_dict.items(), columns=['value', 'cnt'])
        val_cnt_df.sort_values(by=['cnt', 'value'], inplace=True, ascending=False)
        val_cnt_df.to_csv(out_path, index=False)
        return val_cnt_df

    @staticmethod
    def dump_html_dir(csv_dir, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in csv_dir.glob('**/*.csv'):
            out_path = out_dir / csv_path.name.replace('.csv', '.html')
            pd.read_csv(csv_path).to_html(out_path)

    @staticmethod
    def val_cnt_cols(df):
        n_row = min(df.shape[0], 100)
        df = df[:n_row]
        cols = []
        for col in df.columns:
            try:
                df[col].value_counts()
                cols.append(col)
            except:
                print('skip: {}'.format(col))
        return cols

    @staticmethod
    def get_val_cnts(df, col):
        val_cnts = df[col].value_counts(dropna=False).rename_axis('value').reset_index(name='cnt')
        val_cnts.sort_values(by=['cnt'], ascending=False, inplace=True)
        return val_cnts

    @staticmethod
    def dump_df_stats(df, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        val_cnt_cols = Util.val_cnt_cols(df)
        for col in val_cnt_cols:
            val_cnts = Util.get_val_cnts(df, col)
            val_cnts.to_csv(out_dir / (str(col) + '.csv'), index=False)
        describe_all = df[val_cnt_cols].describe(include='all')
        describe_all.to_csv(out_dir / 'describe__all.csv')

    @staticmethod
    def infer_dtype(col_series):
        if col_series.nunique() == 1 and col_series.unique()[0] == '':
            return object
        try:
            return pd.to_numeric(col_series, errors='raise').dtype
        except ValueError:
            return object

    @staticmethod
    def dump_list(data, fpath):
        with open(fpath, "w") as f:
            for elem in data:
                f.write(str(elem) + "\n")

    @staticmethod
    def load_list(fpath):
        data = []
        with open(fpath, "r") as f:
            for line in f:
                data.append(int(line.strip()))
        return data

    @staticmethod
    def cnt_lines(jsonl_path):
        cnt = 0
        with open(jsonl_path) as f:
            for k, line in enumerate(f):
                if k % 1000000 == 0:
                    print(k)
                cnt += 1
        return cnt

    @staticmethod
    def load_dict(jsonl_path, key):
        print('loading {},,'.format(jsonl_path))
        data = {}
        with open(jsonl_path) as f:
            for line in f:
                line = json.loads(line)
                data[line[key]] = line
        return data

    @staticmethod
    def load_lines(jsonl_path, n_rows=-1):
        print('loading: {}'.format(jsonl_path))
        data = []
        with open(jsonl_path) as f:
            for k, line in enumerate(f):
                if k % 1000000 == 0:
                    print(k)
                if n_rows != -1 and k == n_rows:
                    return data
                data.append(json.loads(line))
        return data

    @staticmethod
    def dump_lines(lines, out_path):
        with open(out_path, 'w') as fout:
            for line in lines:
                fout.write(json.dumps(line)+'\n')

    @staticmethod
    def shuf(lines):
        random.shuffle(lines)
        return lines

    @staticmethod
    def get_uniq_header(jsonl_path):
        header_cnt = {}
        header_comb_cnt = {}
        with open(jsonl_path) as f:
            for line in f:
                line = json.loads(line)
                header = line['table_data'][0]
                header_comb = ','.join(header).lower()
                header_comb_cnt[header_comb] = header_comb_cnt.get(header_comb, 0) + 1
                for elem in header:
                    header_cnt[elem.lower()] = header_cnt.get(elem.lower(), 0) + 1
        header_cnt = pd.DataFrame(header_cnt.items(), columns=['value', 'cnt'])
        header_cnt.sort_values(by=['cnt'], ascending=False, inplace=True)
        return header_cnt, header_comb_cnt

    @staticmethod
    def get_cell_cnt(lines):
        uniq_cell_cnt = {}
        for k, line in enumerate(lines):
            for row in line['table_data']:
                for elem in row:
                    uniq_cell_cnt[elem] = uniq_cell_cnt.get(elem, 0) + 1
        cell_cnt = pd.DataFrame(uniq_cell_cnt.items(), columns=['val', 'cnt'])
        cell_cnt.sort_values(by=['cnt'], ascending=False, inplace=True)
        return cell_cnt

    @staticmethod
    def _add_coltype(lines):
        for k, line in enumerate(lines):
            table_data = np.array(line['table_data'])
            table_data = pd.DataFrame(table_data[1:, :], columns=table_data[0, :])
            lines[k]['dtypes'] = []
            for col_idx in range(table_data.shape[1]):
                dtype = Util.infer_dtype(table_data.iloc[:, col_idx])
                if 'object' in str(dtype):
                    lines[k]['dtypes'].append('text')
                elif 'int' in str(dtype) or 'float' in str(dtype):
                    lines[k]['dtypes'].append('real')
                else:
                    ValueError('invalid coltype')
        return lines

    @staticmethod
    def add_coltype(input_fpath, out_fpath):
        data = Util.load_lines(input_fpath)
        out_data = Util._add_coltype(data)
        Util.dump_lines(out_data, out_fpath)

    @staticmethod
    def to_dict(data_list):
        data_dict = {}
        for elem in data_list:
            data_dict[elem] = ''
        return data_dict

    @staticmethod
    def dump_html(df, out_path):
        html_str = df.to_html(escape=False, header=True, index=True, max_rows=None)
        with open(out_path, "w") as fout:
            fout.write(html_str)
            fout.write('<p><a href="./">back</a></p>')

    @staticmethod
    def csv_to_html(csv_path, out_path):
        data = pd.read_csv(csv_path)
        Util.dump_html(data, out_path)

    @staticmethod
    def _add_header_pred(line, prob_h):
        for j, header in enumerate(line['header']):
            line['header'][j] = header + '<br><p></p>' + str(np.round(prob_h[j][1], 3))
        if line['col_labels'] is not None and line['col_labels'] != 'None':
            for cid in line['col_labels']:
                print(cid)
                line['header'][cid] += '_fake'
        return line['header']

    @staticmethod
    def _add_cell_pred(line, prob_c):
        line['table'] = line['table_data_raw']
        for i, row in enumerate(line['table_data_raw']):
            for j, cell in enumerate(row):
                line['table'][i][j] = cell + '<br><p></p>' + str(np.round(prob_c[i][j][1], 3))
        if line['cell_labels'] is not None and line['cell_labels'] != 'None':
            print(line['cell_labels'])
            for rid, cid in line['cell_labels']:
                line['table'][rid][cid] += '_fake'
        return line['table']

    @staticmethod
    def add_pred(lines):
        for k, line in enumerate(lines):
            line['id'] = line['id'].replace('/', '_').replace('.txt', '')
            line['header'] = Util._add_header_pred(line, np.array(line['prob_headers']))
            line['table_data'] = Util._add_cell_pred(line, np.array(line['prob_cells']))
            line['table_data'] = np.vstack([line['header'], line['table_data']])
        return lines

    @staticmethod
    def ft_jsonl_to_html(lines, out_dir_base):
        lines = Util.add_pred(lines)
        dir_cnt = 0
        for i, line in enumerate(lines):
            out_dir = out_dir_base / str(dir_cnt)
            out_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_dir / '{}.html'.format(line['id'].replace(' ', '').replace('/', '__')[:100])

            key = 'table_data' if 'table_data' in line else 'table'
            # html_str = pd.DataFrame(line[key]).to_html(escape=False, header=True, index=True, max_rows=None)
            table = pd.DataFrame(line[key])
            pd.set_option('display.max_colwidth', 100)
            html_str = (table.style.applymap(Util._add_color).set_table_attributes('border="1" class="dataframe table table-hover table-bordered"').render())
            with open(out_path, 'w') as fout:
                fout.write(html_str)
                fout.write('<p><a href="./">back</a></p>')
                fout.write('<p></p><p></p>')
            if (i+1) % 100000 == 0:
                dir_cnt += 1
        print(out_dir_base)

    @staticmethod
    def jsonl_to_html(lines, out_dir_base):
        dir_cnt = 0
        for i, line in enumerate(lines):
            out_dir = out_dir_base / str(dir_cnt)
            out_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_dir / '{}.html'.format(line['id'].replace(' ', '').replace('/', '__')[:100])

            key = 'table_data' if 'table_data' in line else 'table'
            html_str = pd.DataFrame(line[key]).to_html(escape=False, header=True, index=True, max_rows=None)
            with open(out_path, 'w') as fout:
                fout.write(html_str)
                fout.write('<p><a href="./">back</a></p>')
                fout.write('<p></p><p></p>')
                for k, v in line.items():
                    fout.write('{}: {}'.format(k, v))
                    fout.write('<p></p><p></p>')
            if (i+1) % 100000 == 0:
                dir_cnt += 1
        print('dump html on {}'.format(out_dir_base.absolute()))

    @staticmethod
    def to_html(jsonl_path, outdir_path):
        lines = Util.load_lines(jsonl_path)
        Util.jsonl_to_html(lines, outdir_path)

    @staticmethod
    def print_jsonl(jsonl_path):
        lines = Util.load_lines(jsonl_path)
        for line in lines:
            for k, v in line.items():
                print(k, v)
            print()

    @staticmethod
    def get_uniq_cells(input_jsonl, max_row=1000, max_col=10000):
        keys = {}
        with open(input_jsonl) as f:
            for k, line in enumerate(f):
                if k % 100000 == 0:
                    print(k)
                line = json.loads(line)
                uniq_vals = np.unique(np.array(line['table_data'])[:max_row, :max_col])
                for v in uniq_vals:
                    keys[v.lower()] = 1
        return list(keys.keys())

    @staticmethod
    def get_label_type(label_path):
        label = pd.read_csv(label_path)
        label = label.columns.tolist()
        ltype = ''
        if 'row_id' in label and 'col_id' in label:
            ltype = 'cell'
        elif 'row_id' not in label and 'col_id' in label:
            ltype = 'col'
        elif 'row_id' not in label and 'col_id' in label and 'label' in label:
            ltype = 'table'
        return ltype

    @staticmethod
    def to_label_dict(label_path, label_type):
        label = pd.read_csv(label_path)
        label_dict = {}
        for _, row in label.iterrows():
            fid = row['fname'].split('.csv')[0]
            label_dict[fid] = label_dict.get(fid, [])
            if label_type == 'cell':
                label_dict[fid].append([row['row_id'], row['col_id']])
            elif label_type == 'col':
                label_dict[fid].append(row['col_id'])
            elif label_type == 'table':
                label_dict[fid].append(row['label'])
        return label_dict

    @staticmethod
    def csvdir_to_jsonl(csv_dir, out_jsonl_path, label_path=None, label_type=None):
        new_lines = []
        if label_path is not None:
            label_dict = Util.to_label_dict(label_path, label_type=label_type)
        for fpath in Path(csv_dir).glob('*.csv'):
            table = pd.read_csv(fpath, header=None, keep_default_na=False, dtype=str).replace(np.nan, '', regex=True).values
            line = {'id': fpath.stem, 'table_data': table.tolist()}
            if label_path is not None:
                line['{}_labels'.format(label_type)] = label_dict[line['id']] if line['id'] in label_dict else []
            new_lines.append(line)
        Util.dump_lines(new_lines, out_jsonl_path)

    @staticmethod
    def to_fake_header(table, fake_headers):
        ftable = np.array(copy.deepcopy(table), dtype=object)
        for cid, word in fake_headers:
            ftable[0, cid] = word
        return ftable.tolist()

    @staticmethod
    def to_fake_cell(table, fake_cells):
        ftable = np.array(copy.deepcopy(table), dtype=object)
        for rid, cid, word in fake_cells:
            ftable[rid+1, cid] = word
        return ftable.tolist()

    @staticmethod
    def _add_color(val):
        if val is not None:
            color = 'red' if ('fake' in val) else 'black'
            if '_high' in val:
                color = 'blue'
        else:
            color = 'black'
        return 'color: %s' % color

    @staticmethod
    def add_labels(jsonl_path, out_path, label_type):
        fheaders, fcells = False, False
        if label_type == 'cell':
            fcells = True
        elif label_type == 'col':
            fheaders = True

        lines = Util.load_lines(jsonl_path)
        new_lines = []
        for line in lines:
            table = copy.deepcopy(line['table_data'])
            if fheaders:
                table = Util.to_fake_header(table, line['faked_headers'])
            if fcells:
                table = Util.to_fake_cell(table, line['faked_cells'])

            new_line = {
                'id': copy.deepcopy(line['id']),
                'table_data': table,
            }
            if label_type == 'cell':
                new_line['cell_labels'] = [[fake_cell[0], fake_cell[1]] for fake_cell in line['faked_cells']]
            elif label_type == 'col':
                new_line['col_labels'] = [fake_header[0] for fake_header in line['faked_headers']]
            elif label_type == 'table':
                new_line['table_labels'] = int(np.array(line['table_data']).size > 30)
            new_lines.append(new_line)
        Util.dump_lines(new_lines, out_path)

    @staticmethod
    def dump_cell_label(jsonl_path, label_path):
        lines = Util.load_lines(jsonl_path)
        label_df = []
        for line in lines:
            nrows, ncols = np.array(line['table_data']).shape
            for (rid, cid) in line['cell_labels']:
                label_df.append([line['id']+'.csv', rid, cid])
                if nrows <= rid+1:
                    raise ValueError('invalid row id')
        pd.DataFrame(label_df, columns=['fname', 'row_id', 'col_id']).to_csv(label_path, index=False)

    @staticmethod
    def dump_col_label(jsonl_path, label_path):
        lines = Util.load_lines(jsonl_path)
        label_df = []
        for line in lines:
            for cid in line['col_labels']:
                label_df.append([line['id']+'.csv', cid])
        pd.DataFrame(label_df, columns=['fname', 'col_id']).to_csv(label_path, index=False)

    @staticmethod
    def dump_tab_label(jsonl_path, label_path):
        lines = Util.load_lines(jsonl_path)
        label_df = []
        for line in lines:
            label_df.append([line['id']+'.csv', int(line['table_labels'])])
        pd.DataFrame(label_df, columns=['fname', 'label']).to_csv(label_path, index=False)

    @staticmethod
    def jsonl_to_csvdir(jsonl_path, outdir):
        lines = Util.load_lines(jsonl_path)
        for line in lines:
            pd.DataFrame(line['table_data']).to_csv(outdir/(line['id']+'.csv'), index=False, header=None)

    @staticmethod
    def test_eq(jsonl_path1, jsonl_path2):
        lines_dict1 = Util.load_dict(jsonl_path1, 'id')
        lines_dict2 = Util.load_dict(jsonl_path2, 'id')
        for k, v in lines_dict1.items():
            v2 = copy.deepcopy(v['cell_labels'])
            copy.deepcopy(lines_dict2[k]['cell_labels'])
            # if v!=lines_dict2[k]:
            #     print(v)
            #     print(lines_dict2[k])
            del lines_dict2[k]['col_labels']
            del lines_dict2[k]['table_labels']
            print(v==lines_dict2[k])
            if v != lines_dict2[k]:
                print(lines_dict2[k])
                print(v)
                raise ValueError('error')
            # print(v)
            # print(lines_dict2[k])
            # print()

    @staticmethod
    def gen_repo_dataset(input_dir, out_dir):
        # data_dir, label_dir = out_dir/'data_jsonl', out_dir/'label'
        cell_dir, col_dir, tab_dir = out_dir/'ft_cell', out_dir/'ft_col', out_dir/'ft_table'
        # label_dir.mkdir(exist_ok=True, parents=True)
        cell_dir.mkdir(exist_ok=True, parents=True)
        col_dir.mkdir(exist_ok=True, parents=True)
        tab_dir.mkdir(exist_ok=True, parents=True)

        # dump label
        for dset in ['train', 'valid', 'test']:
            Util.add_labels(input_dir/'{}.jsonl'.format(dset), cell_dir/'{}.jsonl'.format(dset), 'cell')
            Util.add_labels(input_dir/'{}.jsonl'.format(dset), col_dir/'{}.jsonl'.format(dset), 'col')
            Util.add_labels(input_dir/'{}.jsonl'.format(dset), tab_dir/'{}.jsonl'.format(dset), 'table')
            Util.dump_cell_label(cell_dir/'{}.jsonl'.format(dset), cell_dir/'{}_label.csv'.format(dset))
            Util.dump_col_label(col_dir/'{}.jsonl'.format(dset), col_dir/'{}_label.csv'.format(dset))
            Util.dump_tab_label(tab_dir/'{}.jsonl'.format(dset), tab_dir/'{}_label.csv'.format(dset))

        # dump csv_dir
        for dset in ['train', 'valid', 'test']:
            for label in ['cell', 'col', 'table']:
                csv_dir = out_dir / 'ft_{}/{}_csv'.format(label, dset)
                csv_dir.mkdir(exist_ok=True, parents=True)
                Util.jsonl_to_csvdir(out_dir/'ft_{}/{}.jsonl'.format(label, dset), csv_dir)
                # Util.csvdir_to_jsonl(csv_dir, data_dir/'{}_debug.jsonl'.format(dset), label_path=label_dir/'{}_cell_label.csv'.format(dset), label_type='cell')
                # Util.test_eq(data_dir/'{}_debug.jsonl'.format(dset), data_dir/'{}.jsonl'.format(dset))
                Util.to_html(out_dir/'ft_{}/{}.jsonl'.format(label, dset), out_dir/'html/{}/{}'.format(label, dset))
            Util.to_html(input_dir / '{}.jsonl'.format(dset), input_dir / 'html/{}'.format(label, dset))


def main():
    base_dir = Path('/mnt/nfs/work1/miyyer/hiida/data/pretrain/data/valid26600K/tabbie_repo')
    out_dir = Path('/home/hiida/tabbie_v0/data')
    out_dir.mkdir(exist_ok=True, parents=True)
    Util.gen_repo_dataset(base_dir/'input', out_dir)
    exit()

