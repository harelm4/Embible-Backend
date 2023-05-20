import itertools
from typing import List

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.standard_model import StandardModel


class single_char_model(StandardModel):

    def predict(self, text: str, min_p: float = 0.5) -> ModelResult:
        """
        main function of this class, genetates predictions
        :param text: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param min_p: minimal value of prediction score
        :return: charModelResult , an object encapsulating list of TextParts
        """
        if '?' not in text:
            return ModelResult([TextPart(text, None)])
        preds_w = super(single_char_model, self).predict(text, min_p)
        pres_only = preds_w.get_only_predictions()
        res = []
        for dict in preds_w:
            if '?' not in dict.text:
                splitted_sent = dict.text.split()
                for word in range(len(splitted_sent)):
                    next_pred = TextPart(splitted_sent[word], None)
                    res.append(next_pred)
            else:
                res.append(dict)
        return ModelResult(res)


    def _get_list_of_str_preds(self, predictions: List[Prediction]) -> List[List[str]]:
        list_of_preds = []
        for l in predictions:
            if l == None:
                continue
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)
        return list_of_preds

    def _get_list_of_scores(self, list_of_preds, pres_only):
        list_of_p = []
        for i in range(len(list_of_preds)):
            p_lst = []
            for j in range(len(list_of_preds[i])):
                p_lst.append(pres_only[i].predictions[j].score)
            list_of_p.append(p_lst)
        return list_of_p
