# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:17:22 2023

@author: itaia
"""
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
        results = []
        for entry_idx, entry in enumerate(data):
            print(entry_idx)
            real_values = entry['missing']
            modelRes = model.predict(entry['text']).get_only_k_predictions(k)
            list_of_preds = self._model_result_to_list_of_preds(modelRes)
            all_words_and_missing_indexes = []
            # getting all the missing indexes that we want to predict (for words only !)
            missing_idxs_full_word_mask = self._get_missing_idxs(entry['text'])
            for pred_idx, preds in enumerate(list_of_preds):
                pred_missing_words = []
                for j in range(len(preds)):
                    # Because we return only the indexes from missing words and in the
                    # verse could be another missings (chars/subword but not full word)
                    # So the length of the list of missing indexes words shorter than the
                    # predictions length. If it correct we finish the process and find all the
                    # missing indexes and their predictions
                    if pred_idx > len(missing_idxs_full_word_mask) - 1:
                        break

                    # could be a situation that the model returns a shorter word than what we want to predict
                    if len(missing_idxs_full_word_mask[pred_idx]) > len(preds[j]):
                        pass
                    else:
                        all_words_and_missing_indexes.append((missing_idxs_full_word_mask[pred_idx], preds[j]))

            # all_words_and_missing_indexes holds lists of tuples: [([4,5,6,7],'עוהב'),([4,5,6,7], 'אוהב')...]
            fit_count = 0
            for i, c_preds in enumerate(all_words_and_missing_indexes):
                flag = True
                for m, mis_index in enumerate(c_preds[0]):

                    # if there is  different between one of the chars from the word we predict to the real word
                    # the flag will be false, else will be true
                    # and add 1 to the amount of words that we corrected
                    if real_values[str(mis_index)] != c_preds[1][m]:
                        flag = False

                if flag == True:
                    fit_count += 1

            if len(all_words_and_missing_indexes) == 0:
                results.append(0)
            else:
                results.append(fit_count / len(all_words_and_missing_indexes))
        return sum(results) / len(results)

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

    def _get_missing_idxs(self, text: str) -> List[int]:
        """
        for word with missing parts at index `pred_idx` ,
        this function returns a list of the indexes that are missing in it relative to the start of the word
        :param pred_idx: index of a word with missing parts (out of just the missing words)
        :param text: the input text of the prediction
        :return: list of missing chars indexes
        """
        counter_indexes = 0
        list_indexes = []
        split = text.split(' ')

        for i, word in enumerate(split):
            if word.count('?') < len(word):
                counter_indexes += len(word)

            if word.count('?') == len(word):
                lst_indexes_word = [*range(counter_indexes, counter_indexes + len(word), 1)]
                list_indexes.append(lst_indexes_word)
                counter_indexes += len(word)

            if i != len(split) - 1:
                counter_indexes += 1
        return list_indexes
