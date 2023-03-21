from typing import List

import pandas as pd
from transformers import pipeline

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.model import Model
from src.utils.strings import StringUtils


class SequentialModel(Model):

    def predict(self, text: str, min_p: float = 0.01) -> ModelResult:
        mlm = pipeline("fill-mask", model=self.model_path)
        if '?' not in text:
            return ModelResult([TextPart(text, None)])
        preds = self._predict_all(mlm, text)
        splited_text = StringUtils.split_text(text)

        res = []
        pres_index = 0
        for i, part in enumerate(splited_text):
            if '?' in part:
                res.append(TextPart('?', [Prediction(x[0], x[1]) for x in preds[pres_index] if x[1] >= min_p]))
                pres_index += 1
            else:
                res.append(TextPart(part, None))

        return ModelResult(res)

    def _predict_all(self, mlm, txt):
        count = txt.count('?')
        input_text = StringUtils.insert_masks(txt)
        all_preds = []
        for i in range(count):
            preds = mlm(input_text)
            try:
                input_text = input_text.replace('[MASK]', preds[0][0]['token_str'], 1)
                for pred in preds[0]:
                    all_preds.append({'token_index': i, 'token': pred['token_str'], 'score': pred['score']})
            except:
                input_text = input_text.replace('[MASK]', preds[0]['token_str'], 1)
                pred = preds[0]
                all_preds.append({'token_index': i, 'token': pred['token_str'], 'score': pred['score']})

        return pd.DataFrame(all_preds).groupby('token_index').apply(
            lambda x: list(zip(x['token'], round(x['score'], 2)))).to_dict()
