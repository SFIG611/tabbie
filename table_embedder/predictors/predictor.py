
import os
import json
import numpy as np
from pathlib import Path
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('predictor')
class PretrainPredictor(Predictor):
    @staticmethod
    def get_fake_vals(line):
        fake_headers, fake_cell_ids, fake_cell_words = None, None, None
        if 'faked_headers' in line:
            fake_headers = line['faked_headers']
        if 'faked_cells' in line:
            fake_cell_ids = [[fake_cell[0], fake_cell[1]] for fake_cell in line['faked_cells']]
            fake_cell_words = [fake_cell[2] for fake_cell in line['faked_cells']]
        return fake_headers, fake_cell_ids, fake_cell_words

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        line = json_dict
        table = line['table_data']
        idx = line['old_id'] if 'old_id' in line else line['id']

        fake_headers, fake_cell_ids, fake_cell_words = self.get_fake_vals(line)
        max_rows, max_cols = 30, 20
        if len(table[0]) > max_cols:
            table = np.array(table)[:, :max_cols].tolist()

        table = table[0:max_rows]
        table_np = np.array(table)
        blank_loc = np.argwhere((table_np == '') | (table_np == '-') | (table_np == 'n/a') | (table_np == '&nbsp;'))

        instance = self._dataset_reader.text_to_instance(
            id=idx,
            header=table[0],
            table=table[1:max_rows],
            blank_loc=blank_loc,
            replace_cell_ids=fake_cell_ids,
            replace_cell_words=fake_cell_words,
            replace_headers=fake_headers,
        )
        return instance





