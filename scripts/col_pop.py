
import os
import copy
import json
import subprocess

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.util import Util


class ColPopEval:
    @staticmethod
    def evaluate(pred_fpath, out_fpath):
        pred_results = []
        with open(pred_fpath) as f:
            for k, line in enumerate(f):
                line = json.loads(line)
                for j, pred_label_name in enumerate(line['pred_labels_name']):
                    pred_results.append([k+1, 'Q0', pred_label_name.replace(' ', '_')[:100], j+1, 1/(j+1), 'labelrun1'])
        result_df = pd.DataFrame(np.array(pred_results), columns=['tid', 'Q0', 'label', 'rank', 'score', 'labelrun1'])
        result_df.drop_duplicates(['tid', 'label'], inplace=True)
        result_df['score'] = result_df['score'].astype(np.float64)
        result_df['tid'] = result_df['tid'].astype(int)
        result_df.to_csv(out_fpath, index=False, sep="\t", header=False)

    @staticmethod
    def eval_all(base_dir, gt_dir):
        print(base_dir)
        for fname in ['cp_result1.jsonl', 'cp_result2.jsonl', 'cp_result3.jsonl']:
            assert os.path.exists(base_dir/fname)
        for i in range(1, 4):
            ColPopEval.evaluate(base_dir/'cp_result{}.jsonl'.format(i), base_dir/'cp_R{}.txt'.format(i))
        for i in range(1, 4):
            cmd = ['/home/hiida/trec_eval/trec_eval', str(gt_dir/'gt_c{}.txt'.format(i)), base_dir/'cp_R{}.txt'.format(i)]
            result = subprocess.run(cmd)
            print(result)
            cmd = ['/home/hiida/trec_eval/trec_eval', '-m', 'ndcg_cut', str(gt_dir/'gt_c{}.txt'.format(i)), base_dir/'cp_R{}.txt'.format(i)]
            result = subprocess.run(cmd)
            print(result)


class ColPopDev:
    @staticmethod
    def debug(base_dir):
        lines = Util.load_lines(base_dir/'valid_nolabel.jsonl')
        for line in lines:
            print(line['table_data'])
            exit()
        exit()


class ColPop:
    @staticmethod
    def _dump_header_cnt(jsonl_path, out_path):
        header_cnt, _ = Util.get_uniq_header(jsonl_path)
        header_cnt['value'] = header_cnt['value'].str.lower()
        header_cnt.to_csv(out_path, index=False)

    @staticmethod
    def _dump_baseline_header(baseline_path, out_path):
        baseline = pd.read_csv(baseline_path, delimiter='\t', header=None)
        val_cnt = Util.get_val_cnts(baseline, 2)  # 2: column name
        val_cnt.to_csv(out_path, index=False)

    @staticmethod
    def _get_max_label(json_path):
        labels = []
        with open(json_path) as f:
            for line in f:
                line = json.loads(line)
                for label_idx in line['label_idx']:
                    labels.append(label_idx)
        labels.sort()
        return max(labels)

    @staticmethod
    def _add_label(tables_path, n_seed_cols, label, out_path, fix_seed=True):
        fout = open(out_path, 'w')
        f = open(tables_path)
        for orig_line in f:
            line = json.loads(copy.deepcopy(orig_line))
            table_data = np.array(line['table_data'])
            if len(table_data[0]) <= n_seed_cols:
                print('skip: {}'.format(line['id']))
                continue
            n_seed = copy.deepcopy(n_seed_cols)
            if not fix_seed:
                n_seed = np.random.randint(0, n_seed_cols) + 1
            one_row = copy.deepcopy(line)
            one_row['orig_header'] = table_data[0].tolist()
            one_row['label_idx'], one_row['label'] = [], []
            for i in range(n_seed, len(table_data[0])):
                header_elem = table_data[0, i].lower()
                one_row['label'].append(header_elem)
                one_row['label_idx'].append(label[header_elem])
            one_row['table_data'] = (np.array(line['table_data'])[:, :n_seed]).tolist()
            fout.write(json.dumps(one_row)+'\n')
        f.close()
        fout.close()

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
        return label

    @staticmethod
    def dump_labels(out_dir, wiki_path, baseline_r3_path, combined_r3_path):
        out_dir = Path(out_dir)
        for prefix in ['valid', 'test']:
            ColPop._dump_header_cnt(out_dir/'{}_nolabel.jsonl'.format(prefix), out_dir/'{}_tables_header.csv'.format(prefix))
        ColPop._dump_header_cnt(wiki_path, out_dir/'wikitables_header.csv')
        ColPop._dump_header_cnt(out_dir/'train_nolabel.jsonl', out_dir/'label.csv')
        ColPop._dump_baseline_header(baseline_r3_path, out_dir/'baseline_R3_header.csv')
        ColPop._dump_baseline_header(combined_r3_path, out_dir/'combined_R3_header.csv')

    @staticmethod
    def mod_test(test_path, out_test_path):
        lines = Util.load_lines(test_path)
        for k, line in enumerate(lines):
            table = np.array(line['table_data'])
            if table[0][0] != '':
                continue
            table = table[:, 1:]
            print(line['id'])
            lines[k]['table_data'] = table.tolist()
        Util.dump_lines(lines, out_test_path)

    @staticmethod
    def add_labels(label_path, out_dir, train_path, valid_path, test_path):
        out_dir = Path(out_dir)
        label_map = ColPop.load_label(label_path)
        # ColPop._add_label(train_path, 3, label_map, out_dir/'train_all.jsonl', fix_seed=False)
        # ColPop._add_label(valid_path, 3, label_map, out_dir/'valid_all.jsonl', fix_seed=False)
        # train = Util.load_list(out_dir/'train_all.jsonl')
        # train = Util.shuffle(train)
        # Util.dump_list(train, out_dir/'train_shuf.jsonl')
        # Util.dump_list(train[:100000], out_dir/'train_100K_v2.jsonl')
        out_test_path = './cp/tmp/test.jsonl'
        ColPop.mod_test(test_path, out_test_path)

        for i in range(1, 4):  # 1-4: seed n_cols
            # Preprocess._add_label(train_tables_path, i, label_map, out_dir/'train{}.jsonl'.format(str(i)))
            ColPop._add_label(out_test_path, i, label_map, out_dir/'test{}.jsonl'.format(str(i)))

    @staticmethod
    def _prepro_line(line):
        for i, row in enumerate(line['table_data']):
            for j, elem in enumerate(row):
                line['table_data'][i][j] = Util.remove_bracket(elem, 2)
                # if elem != line['table_data'][i][j]:
                #     line['table_data'][i][j] = line['table_data'][i][j].replace('_', ' ')
        return line

    @staticmethod
    def preprocess(out_dir):
        all_lines = []
        for prefix in ['train', 'valid', 'test']:
            lines = Util.load_lines(Path(out_dir)/'{}_tables.jsonl'.format(prefix))
            new_lines = []
            for line in lines:
                line = ColPop._prepro_line(line)
                new_lines.append(line)
                all_lines.append(line)
            Util.dump_lines(new_lines, Path(out_dir)/'{}_nolabel.jsonl'.format(prefix))
        Util.dump_lines(all_lines, Path(out_dir)/'all_nolabel.jsonl')

    @staticmethod
    def del_empty_label(fpath, out_path):
        lines = Util.load_lines(fpath)
        new_lines = []
        cnt = 0
        for line in lines:
            new_labels = []
            new_label_idx = []
            for k, label in enumerate(line['label']):
                if label == '':
                    print('skip: {}'.format(cnt))
                    cnt+=1
                    continue
                new_labels.append(label)
                new_label_idx.append(line['label_idx'])
            line['label'] = new_labels
            line['label_idx'] = new_label_idx
            new_lines.append(line)
        Util.dump_lines(lines, out_path)

    @staticmethod
    def view_data():
        label_path = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/label.csv')
        label = pd.read_csv(label_path)
        # print(label)
        # exit()
        # fpath = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/valid_types.jsonl')
        fpath0 = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/train_all_100K.jsonl')
        fpath1 = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/test1_types.jsonl')
        fpath2 = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/test2_types.jsonl')
        fpath3 = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/test3_types.jsonl')
        with open(fpath0) as f:
            for line in f:
                line = json.loads(line)
                ok = False
                for lid in line['label_idx']:
                    if int(lid) < 127656:
                        ok = True
                        continue
                if not ok:
                    print(line['label_idx'])
                    print(line['label'])
                    print()
                # print(len(line['table_data'][0]))
        exit()


def main():
    f = Util.load_yaml('./setting/col_pop.yml')
    # ColPopDev.debug(Path(f['out_dir']))
    # exit()
    # TabEL.dump_wiki(f['wiki'], f['valid_id'], f['test_id'], f['train'], f['valid'], f['test'])
    # ColPop.preprocess(f['out_dir'])
    # ColPop.dump_labels(f['out_dir'], f['wiki'], f['baseline'], f['combined'])
    # ColPop.add_labels(f['label'], f['out_dir'], f['train'], f['valid'], f['test'])
    # exit()
    # ColPop.view_data()
    # print('test')
    # exit()
    # base_dir = Path('/mnt/nfs/work1/miyyer/hiida/data/col_pop/data/')
    # ColPop.del_empty_label(base_dir/'test1_types.jsonl', base_dir/'test1_fix.jsonl')
    # exit()

    # base_dir = Path('../exp/cp/exp_result/3M5e_mix50')
    # base_dir = Path('../exp/cp/exp_result/3M5e_mix125')
    # base_dir = Path('/home/hiida/code/table_emb/exp/cp/exp_result/26600K453e_bs24_v2/')
    # base_dir = Path('/home/hiida/code/table_emb/exp/cp/exp_result/26600K453e_bs24_nrow10/')
    # base_dir = Path('/home/hiida/code/table_emb/exp/cp/exp_result/26600K453e_bs12_v4/')
    base_dir = Path('/home/hiida/code/table_emb/exp/cp/exp_result/26600K436e_bs12_v4/test_mod')
    # base_dir = Path('/home/hiida/code/table_emb/exp/cp/exp_result/26600K453e_bs12_v4/test_mod')
    gt_dir = Path('../exp/cp/data/gt')
    ColPopEval.eval_all(base_dir, gt_dir)
    # pred_path = '../exp/cp/exp_result/0913_5M/cp_result3.jsonl'
    # out_fpath = '../exp/cp/exp_result/0913_5M/cp_R3.jsonl'
    # ColPopEval.evaluate(pred_path, out_fpath)


if __name__ == '__main__':
    main()


