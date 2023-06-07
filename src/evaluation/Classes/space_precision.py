from typing import List

import pandas as pd
from tqdm import tqdm
from src.space_predictor.space_predictor import space_predictor


class space_precision():

    def calculate(self,predictor:space_predictor,data: str or List[dict],min_p)->float:
        if isinstance(data, str):
            data = self.get_data_at_test_format(data)
        tp=0
        fp=0
        progress_bar = tqdm(range(len(data)), desc=f"{predictor.__class__.__name__} SpacePrecision", unit="entry",
                            bar_format="\033[32m{l_bar}{bar}{r_bar}\033[0m")
        for entry_idx, entry in enumerate(data):
            progress_bar.update(1)
            text=entry["text"]
            missing=entry["missing"]

            res_text = predictor.genText(text,min_p)
            for i,v in missing.items():
                if res_text[int(i)]==' ' and v==' ':
                    tp+=1
                elif res_text[int(i)]==' ' and v!=' ':
                    fp+=1
        return tp/(tp+fp)




    def get_data_at_test_format(self,file_path: str)->list:
        """
        A function that coverts data from data foldr format to the formerly used format
        :param file_path: test file path
        :return: list in form of [{"text": "...", "missing": {...}}]
        """
        df = pd.read_json(file_path, orient='records', lines=True)
        data = df.to_dict(orient='records')
        return [{'text': x['verse'], 'missing': x['missing_dictionary']} for x in data]

