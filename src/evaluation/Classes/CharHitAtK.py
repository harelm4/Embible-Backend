import math
from typing import List

from src.classes.model_result import ModelResult
from src.classes.text_part import TextPart
from src.evaluation.Classes.HitAtK import HitAtK
from src.model.model import Model

class CharHitAtK(HitAtK):

    def calculate(self, model: Model, data: str or List[dict], k: int) -> int:
        """
        calculate hit@k score for chars.
        :param model: model to be tested
        :param data: data to be tested on. could be string if its a path to test json file or dict if its a ready to go
                     list in form of [{"text": "...", "missing": {...}}]
        :param k: the k of hit@k
        :return: hit@k score (# of char hists at k)/(# of missing chars)
        """
        if type(data) == 'str':
            data = self.get_data_at_hit_at_k_test_format(data)

        for entry_idx, entry in enumerate(data):
            real_values = list(entry['missing'].values())
            modelRes = model.predict(entry['text']).get_only_k_predictions(k)
            list_of_preds = self._model_result_to_list_of_preds(modelRes)
            char_lst = []
            for pred_idx, preds in enumerate(list_of_preds):
                missing_idxs = self._get_missing_idxs(pred_idx, entry['text'])
                for j in missing_idxs:
                    pred_missing_chars = []
                    for w in preds:
                        try:
                            pred_missing_chars.append(w[j])
                        except:
                            # its possible that the model will predict a shorter
                            # word then expected and then index j would be out of range
                            pass
                    char_lst.append(pred_missing_chars)

            fit_count = 0
            for i, c_preds in enumerate(char_lst):
                if real_values[i] in c_preds:
                    fit_count += 1

            return fit_count / len(real_values)

    

    def _get_missing_idxs(self, pred_idx: int, text: str) -> List[int]:
        """
        for word with missing parts at index `pred_idx` ,
        this function returns a list of the indexes that are missing in it relative to the start of the word
        :param pred_idx: index of a word with missing parts (out of just the missing words)
        :param text: the input text of the prediction
        :return: list of missing chars indexes
        """
        missing_word_count = -1
        split = text.split(' ')
        for i, word in enumerate(split):
            if '?' in word:
                missing_word_count += 1
                if pred_idx == missing_word_count:
                    return [c_i for c_i, c in enumerate(word) if c == '?']
