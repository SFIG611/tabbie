
import copy
import os
import json
import numpy as np
from pathlib import Path
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('cell_predictor')
class CellPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        line = json_dict
        idx = line['old_id'] if 'old_id' in line.keys() else line['id']

        max_rows, max_cols = 30, 20
        table_data = line['table_data']
        table = copy.deepcopy(table_data)
        table_header = table_data[0]
        table_data = table_data[1:max_rows]
        if len(table_data[0]) > max_cols:
            table_header = np.array(table_header)[:max_cols].tolist()
            table_data = np.array(table_data)[:, :max_cols].tolist()

        cell_labels = line['cell_labels'] if 'cell_labels' in line else None
        col_labels = line['col_labels'] if 'col_labels' in line else None
        table_labels = line['table_labels'] if 'table_labels' in line else None

        instance = self._dataset_reader.text_to_instance(
            table_id=idx,
            header=table_header,
            cell_labels=cell_labels,
            col_labels=col_labels,
            table_labels=table_labels,
            table_data=table_data,
            table=table
        )
        return instance





