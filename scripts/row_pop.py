
import re
import json
import copy
import shutil
import collections

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.util import Util
# from tabEL import TabEL


class RowPopDev:
    @staticmethod
    def _dump_over3_gt(gt, label):
        pass

    @staticmethod
    def _dump_over3_pred(pred, label):
        pass

    @staticmethod
    def dump_over3_gt(gt_path, label_path, out_path, min_cnt=3):
        # load
        label = RowPop.load_label(label_path, key='cnt')
        pred = pd.read_csv(gt_path, delimiter='\t', header=None)

        # filter
        pred['cnt'] = pred[2].map(label)
        pred = pred[pred['cnt']>=min_cnt]
        pred.reset_index(drop=True, inplace=True)

        pred.drop(['cnt'], inplace=True, axis=1)
        pred.to_csv(out_path, sep='\t', index=False, header=False)
        print(pred)

    @staticmethod
    def dump_over3_pred(pred_path, label_path, out_path, min_cnt=3):
        # load
        label = RowPop.load_label(label_path, key='cnt')
        pred = pd.read_csv(pred_path, delimiter='\t', header=None)

        # filter
        pred['cnt'] = pred[2].map(label)
        pred = pred[pred['cnt']>=min_cnt]
        pred.reset_index(drop=True, inplace=True)

        # updte rank
        pred[3] = pred.groupby(0)[4].rank('first', ascending=False).astype(int)
        pred.reset_index(drop=True, inplace=True)
        pred.drop(['cnt'], inplace=True, axis=1)
        pred.to_csv(out_path, sep='\t', index=False, header=False)
        exit()

    @staticmethod
    def compare_over3():
        base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng')
        RowPopDev.dump_over3_gt(base_dir/'gt/gt_e2.txt', base_dir/'row_pop/label.csv', base_dir/'gt/gt_e2_over2.txt')
        RowPopDev.dump_over3_pred(base_dir/'runfile/RP_nega29/row_pop_R3.txt', base_dir/'row_pop/label.csv', base_dir/'runfile/RP_nega29/row_pop_R2_over3.txt')

    @staticmethod
    def _dump_cand(cand_path, out_path):
        lines = Util.load_lines(cand_path)
        cands = []
        for k, line in enumerate(lines):
            if 'cands_col0' not in line:
                continue
            for i, (cand, score) in enumerate(sorted(line['cands_col0'].items(), key=lambda x:x[1], reverse=True)):
                cands.append([k+1, 'Q0', cand.lower(), i+1, score, 'labelrun1'])
        cands = pd.DataFrame(cands, columns=['row_id', 'Q0', 'ent', 'rank', 'score_ent', 'labelrun1'])
        cands.drop_duplicates(subset=['ent'], inplace=True)
        cands.to_csv(out_path, sep='\t', index=False, header=False)
        return cands

    @staticmethod
    def to_rank(base_dir):
        cands = RowPopDev._dump_cand(base_dir/'row_pop/test1_idx0_cand_k10_v4.jsonl', base_dir/'row_pop/debug.csv')
        pred = pd.read_csv(base_dir/'runfile/RP_dev/row_pop_R1.txt', delimiter='\t', header=None)
        pred.columns = ['row_id', 'Q0', 'ent', 'rank', 'score', 'labelrun1']
        cands = cands.merge(pred, on=['row_id', 'ent'], how='left')

        cands['score'].fillna(0, inplace=True)
        cands['Q0'].fillna('Q0', inplace=True)
        cands['labelrun1'].fillna('labelrun1', inplace=True)
        cands['rank'] = cands.groupby('row_id')['score'].rank('first', ascending=False)
        cands['rank'] = cands['rank'].astype(int)
        cands.sort_values(by=['row_id', 'rank'], ascending=[True, True], inplace=True)
        cands.drop_duplicates(subset=['ent'], inplace=True)

        add_id = list(set(pred['row_id'].tolist())-set(cands['row_id'].tolist()))
        pred = pred[pred['row_id'].isin(add_id)]
        cands = pd.concat([cands, pred], ignore_index=True)
        cands.sort_values(by=['row_id', 'rank'], ascending=[True, True], inplace=True)
        cands[['row_id', 'Q0', 'ent', 'rank', 'score', 'labelrun1']].to_csv(base_dir/'runfile/RP_dev/row_pop_R1_0812.txt', sep='\t', index=False, header=None)
        print(cands['row_id'].nunique())
        # print(cands)
        # print(base_dir)
        exit()

        # new_df
        idx_cands = {}
        new_lines = []
        for row in np.array(pred).tolist():
            if row[2] not in idx_cands[row[0]-1]:
                row.append(1)
            else:
                row.append(0)
            new_lines.append(row)
        new_df = pd.DataFrame(new_lines, columns=[0, 1, 2, 3, 4, 5, 6])
        new_df.sort_values(by=[0, 6, 3], ascending=True, inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        new_df['idx'] = new_df.index
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_rows', 100)
        new_df.to_csv('/home/ubuntu/cikm2019-table/table2vec-lideng/runfile/RP_dev/row_pop_R1_0811.txt', sep='\t', header=None, index=None)
        print(new_df)
        exit()
        exit()

    @staticmethod
    def add_freq(base_dir):
        label = pd.read_csv(base_dir/'row_pop/label.csv')
        label_dict = dict(zip(label['value'], label['cnt']))
        gt = pd.read_csv(base_dir/'gt/gt_e1.txt', delimiter='\t', header=None)
        gt[4] = 0
        cnt_dict = {}
        for idx, row in gt.iterrows():
            if row[2] not in label_dict:
                cnt_dict[0] = cnt_dict.get(0, 0) + 1
                gt.iloc[idx, 4] = 0
                continue
            cnt_dict[row[2]] = cnt_dict.get(row[2], 0) + 1
            gt.iloc[idx, 4] = label_dict[row[2]]
        # print(gt)
        # gt.to_csv(base_dir/'gt/gt_e1_cnt.txt', sep='\t', index=False)
        # exit()
        # print(sum(cnt_dict.values()))
        cumsum = 0
        for i, (k, v) in enumerate(sorted(cnt_dict.items(), key=lambda x:x[1], reverse=True)):
            cumsum += v
            print(i, k, v, cumsum)

    @staticmethod
    def _mod_label(input_path, out_path):
        lines = Util.load_lines(input_path)
        new_lines = []
        for line in lines:
            new_labels = []
            for label_idx in line['label_idx']:
                if label_idx == -1:
                    new_labels.append(0)
                else:
                    new_labels.append(label_idx)
            line['label_idx'] = new_labels
            new_lines.append(line)
        Util.dump_lines(new_lines, out_path)

    @staticmethod
    def mod_label(base_dir):
        RowPopDev._mod_label(base_dir/'valid.jsonl', base_dir/'valid_idx0.jsonl')
        RowPopDev._mod_label(base_dir/'test.jsonl', base_dir/'test_idx0.jsonl')
        for i in range(1, 6):
            RowPopDev._mod_label(base_dir/'test{}.jsonl'.format(i), base_dir/'test{}_idx0.jsonl'.format(i))

    @staticmethod
    def compare_labels(base_dir):
        label, test_label = pd.read_csv(base_dir/'label.csv'), pd.read_csv(base_dir/'test_label.csv')
        label = dict(zip(label['value'], label['cnt']))
        ent_cnt = {}
        for val in test_label['value'].unique().tolist():
            ent_cnt[val] = label[val] if val in label else 0
        ent_df = Util.dump_val_cnt(ent_cnt, base_dir/'compare.csv')
        test_label = test_label.merge(ent_df, on='value', how='left')
        test_label.columns = ['value', 'test_cnt', 'train_cnt']
        print('num test labels: {}'.format(test_label['test_cnt'].sum()))
        print('not in train: {}'.format(test_label[test_label['train_cnt']==1]['test_cnt'].sum()))

    @staticmethod
    def dev_label(base_dir):
        lines = Util.load_lines(base_dir/'test5.jsonl')
        labels = []
        for k, line in enumerate(lines):
            for label in line['label']:
                labels.append([k+1, label])
        labels = pd.DataFrame(labels, columns=['idx', 'label'])
        labels['idx_str'] = labels['idx'].astype(str)
        labels['key'] = labels[['idx_str', 'label']].apply(lambda x: '-'.join(x), axis=1)

        test_path = '/home/ubuntu/cikm2019-table/table2vec-lideng/gt/gt_e5.txt'
        test = pd.read_csv(test_path, delimiter='\t', header=None)
        test = test[[0,2]]
        test.columns = ['idx', 'label']
        test['idx_str'] = test['idx'].astype(str)
        test['key'] = test[['idx_str', 'label']].apply(lambda x: '-'.join(x), axis=1)

        labels.sort_values(by=['idx', 'label'], inplace=True, ascending=True)
        test.sort_values(by=['idx', 'label'], inplace=True, ascending=True)
        labels.to_csv(base_dir/'tmp1.csv', index=False)
        test.to_csv(base_dir/'tmp2.csv', index=False)
        key1 = labels['key'].tolist()
        key2 = test['key'].tolist()
        for i, elem in enumerate(key1):
            if elem != key2[i]:
                print(elem, key2[i])
        print(test)
        print(labels)
        exit()


class RowPopEval:
    @staticmethod
    def evaluate(pred_fpath, out_fpath):
        pred_results = []
        lines = Util.load_lines(pred_fpath)
        for k, line in enumerate(lines):
            for j, pred_label_name in enumerate(line['pred_labels_name']):
                pred_results.append([k+1, 'Q0', pred_label_name.replace(' ', '_')[:100], j+1, 1/(j+1), 'labelrun1'])
        result_df = pd.DataFrame(np.array(pred_results), columns=['tid', 'Q0', 'label', 'rank', 'score', 'labelrun1'])
        result_df.drop_duplicates(['tid', 'label'], inplace=True)
        result_df['score'] = result_df['score'].astype(np.float64)
        result_df['tid'] = result_df['tid'].astype(int)
        result_df.to_csv(out_fpath, index=False, sep="\t", header=False)


class RowPop:
    @staticmethod
    def _cell_to_entity(cell):
        return cell.lower().replace(' ', '_')

    @staticmethod
    def _prepro_line(line):
        for i, row in enumerate(line['table_data']):
            for j, elem in enumerate(row):
                line['table_data'][i][j] = Util.remove_bracket(elem, 1)
                if elem != line['table_data'][i][j]:
                    line['table_data'][i][j] = line['table_data'][i][j].replace('_', ' ')
        return line

    @staticmethod
    def _get_entity_tables(lines, min_ent=5, min_col=1):
        new_lines = []
        for k, line in enumerate(lines):
            if len(line['table_data'][0]) <= min_col or len(line['data']) == 0:
                continue
            if sum(TabEL.is_entity(row[0]) for row in line['data']) <= min_ent:
                continue
            line = RowPop._prepro_line(line)
            new_lines.append(line)
        return new_lines

    @staticmethod
    def add_coltypes(base_dir):
        for fpath in base_dir.glob('**/*jsonl'):
            Util.add_coltype(fpath, fpath.parent.parent/(fpath.stem+'_dtypes.jsonl'))

    @staticmethod
    def dump_entity_tables(base_dir):
        for prefix in ['train', 'valid', 'test']:
            lines = Util.load_lines(base_dir/'{}_tables.jsonl'.format(prefix))
            new_lines = RowPop._get_entity_tables(lines)
            Util.dump_lines(new_lines, base_dir/'{}_nolabel.jsonl'.format(prefix))

    @staticmethod
    def dump_ent_seq(base_dir):
        for prefix in ['train', 'valid', 'test']:
            lines = Util.load_lines(base_dir/'{}_nolabel.jsonl'.format(prefix))
            ent_seq = []
            for line in lines:
                one_seq = []
                for i, row in enumerate(line['data']):
                    if TabEL.is_entity(row[0]):
                        ent = RowPop._cell_to_entity(line['table_data'][i+1][0])
                        one_seq.append(ent)
                if len(one_seq) > 1:
                    ent_seq.append(one_seq)
            with open(base_dir/'{}_ent_seq.json'.format(prefix), 'w') as fout:
                fout.write(json.dumps(ent_seq))

    @staticmethod
    def _dump_label(lines, out_path, col_idx=0):
        ent_cnt = {}
        for line in lines:
            for i, row in enumerate(line['data']):
                if TabEL.is_entity(row[col_idx]):
                    ent = RowPop._cell_to_entity(line['table_data'][i+1][col_idx])
                    ent_cnt[ent] = ent_cnt.get(ent, 0) + 1
        _ = Util.dump_val_cnt(ent_cnt, out_path)

    @staticmethod
    def dump_labels(base_dir):
        for prefix in ['train', 'valid', 'test']:
            lines = Util.load_lines(base_dir/'{}_nolabel.jsonl'.format(prefix))
            RowPop._dump_label(lines, base_dir/'{}_tmp.csv'.format(prefix))
        shutil.move(base_dir/'train_tmp.csv', base_dir/'label.csv')
        shutil.move(base_dir/'valid_tmp.csv', base_dir/'valid_cnt.csv')
        shutil.move(base_dir/'test_tmp.csv', base_dir/'test_cnt.csv')

    @staticmethod
    def load_label(label_path, key='value'):
        label_df = pd.read_csv(label_path, keep_default_na=False)
        label_df['index'] = label_df.index
        if key == 'value':
            label = dict(zip(label_df['value'], label_df['index']))
            label = {k: v for k, v in sorted(label.items(), key=lambda x: int(x[1]))}
        elif key == 'index':
            label = dict(zip(label_df['index'], label_df['value']))
            label = {k: v for k, v in sorted(label.items(), key=lambda x: int(x[0]))}
        elif key == 'cnt':
            label = dict(zip(label_df['value'], label_df['cnt']))
            label = {k: v for k, v in sorted(label.items(), key=lambda x: int(x[1]))}
        return label

    @staticmethod
    def _add_label(lines, n_seed_orig, label, fix_seed):
        new_lines = []
        for line_orig in lines:
            line = copy.deepcopy(line_orig)
            table = np.array(line['table_data'])
            header, cells = table[0, :], table[1:, :]
            if len(cells) <= n_seed_orig:
                print('skip: {}'.format(line['id']))
                continue
            n_seed = copy.deepcopy(n_seed_orig) if fix_seed else np.random.randint(0, n_seed_orig) + 1
            line['seed_col0'], line['label_col0'] = cells[:n_seed, 0].tolist(), cells[n_seed:, 0].tolist()
            line['label_idx'], line['label'] = [], []
            for i in range(n_seed, len(cells)):
                if not TabEL.is_entity(line['data'][i][0]):
                    continue
                ent = RowPop._cell_to_entity(cells[i, 0])
                if cells[i, 0] not in line['seed_col0'] and ent not in line['label']:
                    line['label'].append(ent)
                    elem = label[ent] if ent in label else -1
                    line['label_idx'].append(elem)
            line['table_data'] = np.vstack([header, cells[:n_seed, :]]).tolist()
            new_lines.append(line)
        return new_lines

    @staticmethod
    def add_labels(out_dir):
        # dump train/valid/test
        label = RowPop.load_label(out_dir/'label.csv')
        for prefix in ['train', 'valid', 'test']:
            lines = Util.load_lines(out_dir/'{}_nolabel.jsonl'.format(prefix))
            if prefix == 'train':
                lines = Util.shuf(lines)
            lines = RowPop._add_label(lines, 3, label, False)  # 5: n_seed
            Util.dump_lines(lines, out_dir/'{}_seed3.jsonl'.format(prefix))

        # dump test1-5
        # test = Util.load_lines(out_dir / 'test_nolabel.jsonl')
        # for i in range(1, 6):  # 1-6: seed n_rows
        #     test_i = RowPop._add_label(test, i, label, True)
        #     Util.dump_lines(test_i, out_dir/'test{}.jsonl'.format(i))

    @staticmethod
    def view_data(valid_path):
        valid = Util.load_lines(valid_path)
        for line in valid:
            for row in line['table_data']:
                for elem in row:
                    if 'amaguchi' in elem:
                        print(elem)


def main():
    f = Util.load_yaml('./setting/row_pop.yml')

    RowPop.view_data('/mnt/nfs/work1/miyyer/hiida/data/rp/valid.jsonl')
    exit()

    # TabEL.dump_wiki(f['wiki'], f['valid_id'], f['test_id'], f['train'], f['valid'], f['test'])
    # RowPop.dump_entity_tables(Path(f['out_dir']))
    # RowPop.dump_labels(Path(f['out_dir']))
    # RowPop.add_labels(Path(f['out_dir']))

    # base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng/row_pop')
    # RowPopDev.mod_label(base_dir)
    # RowPopDev.dev_label(Path(f['out_dir']))
    # RowPopDev.check_label(Path(f['out_dir']))
    # RowPopDev.compare_labels(Path(f['out_dir']))

    # base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng/')
    # RowPopDev.add_freq(base_dir)

    # base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng/row_pop')
    # RowPop.dump_ent_seq(base_dir)

    # RowPopDev.compare_over3()
    # exit()

    # base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng/row_pop')
    # RowPopDev.mod_label(base_dir)

    base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng/row_pop/rp_tabert')
    RowPop.add_coltypes(base_dir)
    exit()

    # base_dir = Path('/home/ubuntu/cikm2019-table/table2vec-lideng/row_pop/rp_tabert')
    # RowPopDev.to_rank(base_dir)


if __name__ == '__main__':
    main()




