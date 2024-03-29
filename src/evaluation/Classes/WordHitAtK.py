# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:17:22 2023

@author: itaia
"""
import math
from typing import List

from tqdm import tqdm

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

        progress_bar = tqdm(range(len(data)), desc=f"{model.model_path} Word Hit@{k}", unit="entry",
                            bar_format="\033[32m{l_bar}{bar}{r_bar}\033[0m")
        total_score = 0
        amount_of_masking = 0
        for entry_idx, entry in enumerate(data):
            progress_bar.update(1)
            text = entry['text']
            real_values = list(entry['missing'].values())
            modelRes = model.predict(text).get_only_k_predictions(k)
            if hasattr(model, 'space_predictor') and model.space_predictor:
                after_sp_text = model.text
            else:
                after_sp_text = text
            indeces_of_missing_words = [i for i in range(len(after_sp_text.split())) if any(c == "?" for c in
                                                                                            after_sp_text.split()[i])]
            miss, hit = self._comp_after_sp_texts(after_sp_text, entry['missing'])
            recreated_text = self.recreate_text(after_sp_text, miss)
            masked_words = [recreated_text.split()[word] for word in range(len(recreated_text)) if
                            word in indeces_of_missing_words]
            list_of_preds = self._model_result_to_list_of_preds(modelRes)

            mone = hit
            mechane = len(list_of_preds)
            for index, preds in enumerate(list_of_preds):
                if masked_words[index] in preds:
                    mone += 1
            amount_of_masking += mechane
            total_score += mone
        return total_score / amount_of_masking

    def _comp_after_sp_texts(self, after_sp_text: str, miss_dict: dict):
        hit = 0
        for k, v in miss_dict.copy().items():
            if v == ' ' and after_sp_text[int(k)] == ' ':
                del miss_dict[k]
                hit += 1
            elif after_sp_text[int(k)] == ' ' and v != ' ':
                del miss_dict[k]
        return miss_dict, hit

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

    def recreate_text(self, text: str, missing: dict):
        new_text = text
        for key, value in missing.items():
            new_text = new_text[:int(key)] + value + new_text[int(key) + 1:]
        return new_text
