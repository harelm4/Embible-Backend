# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:17:22 2023

@author: itaia
"""
import math
from typing import List
from src.evaluation.Classes.HitAtK import HitAtK
from src.model.model import Model
from src.classes.model_result import ModelResult





class MRR_WORDS(HitAtK):

    def calculate(self, model: Model, data: str or List[dict]) -> float:

        """
        ** this MRR metric for words consider the rank of the prediction and divide it by 1. The result will be sum of all the ranking (that divided by 1)
        dividing by num of masking cases.
        For example:
        model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 0.75), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגיות', 1), Prediction('ומאפי', 1)]),]))
        data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
        The right prediction אוהב rank on second place (the prediction sorted by their score) so it will get 0.5.
        For עוגות the right prediction rank on first place so it will get 1.
        We had 2 cases of masking.
        As a result we except to get the result: (0.5+1)/2 = 0.75.

        :param model: model to be tested
        :param data: data to be tested on. could be string if its a path to test json file or dict if its a ready to go
                     list in form of [{"text": "...", "missing": {...}}]
        :return: MRR score (As explained above)
        """

        if type(data) == 'str':
            data = self.get_data_at_hit_at_k_test_format(data)


        total_score=0
        amount_of_masking=0
        for entry_idx, entry in enumerate(data):
            real_values = entry['missing']
            indeces_of_missing_words=[i for i in range(len(entry["text"].split())) if any(c=="?" for c in entry["text"].split()[i])]
            modelRes = model.predict(entry['text']).get_prediction_sorted()
            recreated_text=self.recreate_text(entry['text'],entry['missing'])
            masked_words=[recreated_text.split()[word] for word in range(len(recreated_text)) if word in indeces_of_missing_words]
            list_of_preds,list_dicts_pred_rank = self._model_result_to_list_of_preds(modelRes)
            mone=0
            mechane=len(list_of_preds)
            for index,preds in  enumerate(list_of_preds):
                if masked_words[index] in preds:
                    mone+= 1/list_dicts_pred_rank[index][masked_words[index]]
            amount_of_masking+=mechane
            total_score+=mone
        return total_score/amount_of_masking

    def _model_result_to_list_of_preds(self, modelRes: ModelResult) -> List[List[str]]:
        """
        converts model result to list of list of prediction strings
        :param textparts: list of textpart
        :return: list of prediction strings
        """
        res = []
        list_of_dict_pred_rank=[]
        for textpart in modelRes.lst:
            preds = []
            dict_pred_rank={}
            index_rank=0
            current_score=1.1
            for pred in textpart.predictions:
                if current_score>pred.score:
                    index_rank+=1
                    current_score=pred.score
                preds.append(pred.value)
                dict_pred_rank[pred.value]=index_rank

            list_of_dict_pred_rank.append(dict_pred_rank)
            res.append(preds)
        return res,list_of_dict_pred_rank

    def recreate_text(self,text:str,missing:dict):
        new_text=text
        for key,value in missing.items():
            new_text=new_text[:int(key)]+value+new_text[int(key)+1:]
        return new_text

