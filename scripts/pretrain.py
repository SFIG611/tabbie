
import os
import json
import copy
import random
import numpy as np
import pandas as pd
from scripts.util import Util


class Pretrain:
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
    def dump_html(lines, out_dir_base):
        dir_cnt = 0
        for i, line in enumerate(lines):
            out_dir = out_dir_base / str(dir_cnt)
            out_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_dir / '{}.html'.format(line['id'])

            table = pd.DataFrame(line['table_data'])
            pd.set_option('display.max_colwidth', 100)
            html_str = (table.style.applymap(Pretrain._add_color).set_table_attributes('border="1" class="dataframe table table-hover table-bordered"').render())
            # html_str = pd.DataFrame(line['table_data']).to_html(escape=False, header=True, index=True, max_rows=None)
            with open(out_path, 'w') as fout:
                fout.write(html_str+'<p><a href="./">back</a></p>')
            if (i+1) % 100000 == 0:
                dir_cnt += 1

    @staticmethod
    def _add_header_pred(line, prob_h):
        line_orig = copy.deepcopy(line)
        for j, header in enumerate(line['header']):
            print(prob_h)
            print(header)
            print(line)
            line['header'][j] = header + '<br><p></p>' + str(np.round(prob_h[j][1], 3))
        if isinstance(line['replace_headers'], list):
            for hid, header in line['replace_headers']:
                if header != line_orig['header'][hid]:
                    line['header'][hid] = header + '<br><p></p>' + str(np.round(prob_h[hid][1], 3)) + '_(fake)'
        return line['header']

    @staticmethod
    def _add_cell_pred(line, prob_c):
        line_orig = copy.deepcopy(line)
        line['table'] = line['table_data_raw']
        for i, row in enumerate(line['table_data_raw']):
            for j, cell in enumerate(row):
                line['table'][i][j] = cell + '<br><p></p>' + str(np.round(prob_c[i][j][1], 3))
                if prob_c[i][j][1] > 0.5:
                    line['table'][i][j] += '_high'
        if isinstance(line['replace_cell_ids'], list):
            for k, (rid, cid) in enumerate(line['replace_cell_ids']):
                if line['replace_cell_words'][k] != line_orig['table_data_raw'][rid][cid]:
                    line['table'][rid][cid] = line['replace_cell_words'][k] + '<br><p></p>' + str(np.round(prob_c[rid][cid][1], 3)) + '_(fake)'
        return line['table']

    @staticmethod
    def add_pred(lines):
        for k, line in enumerate(lines):
            line['id'] = line['id'].replace('/', '_').replace('.txt', '')
            line['header'] = Pretrain._add_header_pred(line, np.array(line['prob_headers']))
            line['table_data'] = Pretrain._add_cell_pred(line, np.array(line['prob_cells']))
            line['table_data'] = np.vstack([line['header'], line['table_data']])
        return lines

    @staticmethod
    def dump_pred(pred_path, base_dir):
        # load pred
        lines = Util.load_lines(pred_path)
        lines = Pretrain.add_pred(lines)

        # dump html
        out_dir = base_dir/'out_pred_html'
        Pretrain.dump_html(lines, out_dir)
        Util.make_tarfile(out_dir.parent/(out_dir.stem + '.tar.gz'), out_dir)


def main():
    pass


if __name__ == '__main__':
    main()



# _, _, uniq_path = CellCnt.dump_meta(out_dir, params['train_path'], valid_path=valid_path)
# Pretrain.filter_50M()
# exit()
# Pretrain.view_data()
# Pretrain.split_data()
# Pretrain.eval_all()
# Pretrain.view_all()
# Pretrain.view_all()
# print('test')

# @staticmethod
# def filter_50M():
#     # fpath = '/home/hiida/code/table_emb/exp/train20M/20M_shuf.jsonl'
#     # out_path = '/home/hiida/code/table_emb/exp/train20M/train20M.jsonl'
#     fpath = '/mnt/nfs/scratch1/hiida/ratio065/50M.jsonl'
#     out_path = '/mnt/nfs/scratch1/hiida/ratio065/train50M.jsonl'
#     with open(out_path, 'w') as fout:
#         with open(fpath) as f:
#             for k, line in enumerate(f):
#                 line = json.loads(line)
#                 line['id'] = line['id'] if 'id' in line else line['old_id']
#                 one_row = {'id': line['id'], 'table_data': line['table_data']}
#                 fout.write(json.dumps(one_row)+'\n')
#                 if k % 1000000 == 0:
#                     print(k)

# @staticmethod
# def split_data():
#     # fpath = '/mnt/nfs/work1/miyyer/hiida/data/pretrain/data/0818/0818_5M_id.jsonl'
#     # out_dir = Path('/home/hiida/code/table_emb/exp/table_emb_5M/train_dir')
#     # fpath = '/mnt/nfs/work1/miyyer/hiida/data/pretrain/data/train25M/train25M_shuf.jsonl'
#     # out_dir = Path('/mnt/nfs/work1/miyyer/hiida/data/pretrain/data/train25M/train_dir')
#     fpath = '/mnt/nfs/work1/miyyer/hiida/data/pretrain/data/train26600K/train26600K_shuf.jsonl'
#     out_dir = Path('/mnt/nfs/work1/miyyer/hiida/data/pretrain/data/train26600K/train_dir')
#     out_dir.mkdir(exist_ok=True, parents=True)
#     nlines = 100000
#     f_tmp, fcnt = None, 0
#     with open(fpath) as f:
#         for k, line in enumerate(f):
#             if k % nlines == 0:
#                 if f_tmp:
#                     f_tmp.close()
#                 f_tmp = open(out_dir/'dataset{}.jsonl'.format(fcnt), "w")
#                 fcnt += 1
#             f_tmp.write(line)
#         if f_tmp:
#             f_tmp.close()

# @staticmethod
# def eval(log_dir):
#     results = {}
#     for fpath in log_dir.glob('metrics_epoch*.json'):
#         with open(fpath) as f:
#             epoch = fpath.name.replace('metrics_epoch_', '').replace('.json', '')
#             print(epoch)
#             results[int(epoch)] = json.load(f)
#     metrics = []
#     for k, v in sorted(results.items(), key=lambda x:x[0]):
#         one_metric = [v['training_h_acc'], v['training_c_acc'], v['training_loss']]
#         one_metric = [str(round(acc, 4)) for acc in one_metric]
#         metrics.append(one_metric)
#     return metrics
#     # print(','.join(accs))
#     # print(k, v)

# @staticmethod
# def print_result(result):
#     for i, elem in enumerate(result):
#         print(i, elem[0], elem[1], elem[2])
#     print()

# @staticmethod
# def view_all():
#     stable = Pretrain.eval(Path('/home/hiida/code/table_emb/exp/train20M/schedule_mix'))
#     # unstable = Pretrain.eval(Path('/home/hiida/code/table_emb/exp/train20M/log_pretrain_0930_2080'))
#     # unstable2 = Pretrain.eval(Path('/home/hiida/code/table_emb/exp/train20M/log_pretrain_0929_v2'))
#     Pretrain.print_result(stable)
#     # Pretrain.print_result(unstable)
#     # Pretrain.print_result(unstable2)

# @staticmethod
# def view_data():
#     fpath = Path('/home/hiida/code/table_emb/exp/train20M/split3/dataset0.jsonl')
#     with open(fpath) as f:
#         for k, line in enumerate(f):
#             line = json.loads(line)
#             print(line['table_data'][0])
#             if k==100:
#                 exit()

# @staticmethod
# def dump_uniq_val1M():
#     train_path = '/home/hiida/code/table_emb/exp/valid26600K/train26600K_valid1M.jsonl'
#     out_dir = Path('/home/hiida/code/table_emb/exp/valid26600K')
#     CellCnt.dump_meta(out_dir, train_path)

#     @staticmethod
#     def dump_html_fake(lines, out_dir):
#         # lines = Util.load_lines(jsonl_path)
#         for k, line in enumerate(lines):
#             lines[k]['id'] = line['id'].replace('/', '_').replace('.txt', '')
#             for i, j, word in line['faked_cells']:
#                 if lines[k]['table_data'][i+1][j] != word:
#                     lines[k]['table_data'][i+1][j] = word + '_(fake)'
#             for j, header in line['faked_headers']:
#                 if lines[k]['table_data'][0][j] != header:
#                     lines[k]['table_data'][0][j] = header + '_(fake)'
#         out_dir.mkdir(exist_ok=True, parents=True)
#         Pretrain.dump_html(lines, out_dir)
#         Util.make_tarfile(str(out_dir)+'.tar.gz', out_dir)
#         print('dump {}'.format(str(out_dir)+'.tar.gz'))

#     @staticmethod
#     def dump_html_raw(lines, out_dir):
#         # lines = Util.load_lines(jsonl_path)
#         for k, line in enumerate(lines):
#             lines[k]['id'] = line['id'].replace('/', '_').replace('.txt', '')
#         out_dir.mkdir(exist_ok=True, parents=True)
#         Util.jsonl_to_html(lines, out_dir)
#         Util.make_tarfile(str(out_dir)+'.tar.gz', out_dir)
#         print('dump {}'.format(str(out_dir)+'.tar.gz'))

#     @staticmethod
#     def add_uniq_id(jsonl_path, out_path):
#         assert os.path.exists(out_path)
#         lines = Util.load_lines(jsonl_path)
#         for k, line in enumerate(lines):
#             idx = line['old_id'] if 'old_id' in line else line['id']
#             lines[k]['id'] = '{}__{}'.format(idx, k)
#         Util.dump_lines(lines, out_path)


