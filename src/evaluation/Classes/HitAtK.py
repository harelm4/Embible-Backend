from abc import ABC
from typing import List

import pandas as pd
from src.classes.model_result import ModelResult

from src.model.model import Model


class HitAtK(ABC):

    def calculate(model:Model,data: str or List[dict],k:int)->int:
        pass

    def get_data_at_hit_at_k_test_format(self,file_path: str)->list:
        """
        A function that coverts data from data foldr format to the formerly used format
        :param file_path: test file path
        :return: list in form of [{"text": "...", "missing": {...}}]
        """
        df = pd.read_json(file_path, orient='records', lines=True)
        data = df.to_dict(orient='records')
        return [{'text': x['verse'], 'missing': x['missing_dictionary']} for x in data]
    
    def _model_result_to_list_of_preds(self, modelRes: ModelResult) -> List[List[str]]:
        """
        converts model result to list of list of prediction strings
        :param textparts: list of textpart
        :return: list of prediction strings
        """
        res = []
        for textpart in modelRes.lst:
            preds = []
            for pred in textpart.predictions:
                preds.append(pred.value)
            res.append(preds)
        return res