import math
from typing import List
from src.classes.model_result import ModelResult
from src.classes.text_part import TextPart
from src.evaluation.Classes.HitAtK import HitAtK
from src.model.model import Model


class WordHitAtK(HitAtK):

    def calculate(self, model: Model, data: str or List[dict], k: int) -> float:
        """
        calculate hit@k score for words.
        :param model: model to be tested
        :param data: data to be tested on. could be string if its a path to test json file or dict if its a ready to go
                     list in form of [{"text": "...", "missing": {...}}]
        :param k: the k of hit@k
        :return: hit@k score (# of words hists at k)/(# of missing words)
        """
        if type(data) == 'str':
            data = self.get_data_at_hit_at_k_test_format(data)


        total_score=0
        amount_of_masking=0
        for entry_idx, entry in enumerate(data):
            real_values = entry['missing']
            indeces_of_missing_words=[i for i in range(len(entry["text"].split())) if any(c=="?" for c in entry["text"].split()[i])]
            modelRes = model.predict(entry['text']).get_only_k_predictions(k)
            recreated_text=self.recreate_text(entry['text'],entry['missing'])
            masked_words=[recreated_text.split()[word] for word in range(len(recreated_text)) if word in indeces_of_missing_words]
            list_of_preds = self._model_result_to_list_of_preds(modelRes)

            mone=0
            mechane=len(list_of_preds)
            for index,preds in  enumerate(list_of_preds):
                if masked_words[index] in preds:
                    mone+=1
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
        for textpart in modelRes.lst:
            preds = []
            for pred in textpart.predictions:
                preds.append(pred.value)
            res.append(preds)
        return res

    def recreate_text(self,text:str,missing:dict):
        new_text=text
        for key,value in missing.items():
            new_text=new_text[:int(key)]+value+new_text[int(key)+1:]
        return new_text