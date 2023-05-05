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
            real_values = entry['missing']
            modelRes = model.predict(entry['text']).get_only_k_predictions(k)
            list_of_preds = self._model_result_to_list_of_preds(modelRes)
            all_words_and_missing_indexes = []
            # getting all the missing indexes that we want to predict (for words only !)
            missing_idxs_full_word_mask,lst_true_of_false_contains_only_mask = self._get_missing_idxs(entry['text'])
            index_missing_words=0
            for pred_idx, preds in enumerate(list_of_preds):
                pred_missing_words = []
                indicator=0
                for j in range(len(preds)):

                    # Because we return only the indexes from missing words and in the
                    # verse could be another missings (chars/subword but not full word)
                    # So the length of the list of missing indexes words shorter than the
                    # predictions length. If it correct we finish the process and find all the
                    # missing indexes and their predictions
                    if index_missing_words > len(missing_idxs_full_word_mask) - 1:
                        break

                    # If the words contains only masking ('?)
                    if lst_true_of_false_contains_only_mask[pred_idx]== True:

                        all_words_and_missing_indexes.append((missing_idxs_full_word_mask[index_missing_words], preds[j]))
                        indicator=1
                
                if indicator==1:
                  index_missing_words+=1
                

            # all_words_and_missing_indexes holds lists of tuples: [([4,5,6,7],'עוהב'),([4,5,6,7], 'אוהב')...]
            fit_count = 0
            for i, c_preds in enumerate(all_words_and_missing_indexes):
                if len(c_preds[0]) != len(c_preds[1]):
                  continue
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

    def _get_missing_idxs(self, text: str) -> List[int]:
        """
        Args:
          text(str)
        Returns:
          list of 2 lists- 1st list: The indexes of words that contains only '?'.
                           2nd list: True of False values. True- if the word contains only '?'. False- if the words contain '?' but other chars.
        
        Example:
            For a scentence: '?? wa?t go ????"
            return: [[0,1],[11,12,13,14]] , [True,False, True]
        """
        counter_indexes = 0
        list_indexes = []
        split = text.split(' ')
        #took only the words contains '?'
        result = list(filter(lambda word: '?' in word, split))
        # True or False if the words contains only '?'
        indicator_list_masking_words = [True if set(word) == set('?') else False for word in result]

        for i, word in enumerate(split):
            if word.count('?') < len(word):
                counter_indexes += len(word)

            if word.count('?') == len(word):
                lst_indexes_word = [*range(counter_indexes, counter_indexes + len(word), 1)]
                list_indexes.append(lst_indexes_word)
                counter_indexes += len(word)

            if i != len(split) - 1:
                counter_indexes += 1
        return [list_indexes,indicator_list_masking_words]
                