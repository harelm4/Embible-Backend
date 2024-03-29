import sys
from abc import ABC
from typing import List

import pandas as pd
from tqdm import tqdm

from src.model.model import Model


class HitAtK(ABC):


    def calculate(self,model:Model,data: str or List[dict],k:int,min_p:float=0.5)->float:
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
