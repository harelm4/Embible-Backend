import math
from typing import List

from tqdm import tqdm

from src.classes.model_result import ModelResult
from src.classes.text_part import TextPart
from src.evaluation.Classes.HitAtK import HitAtK
from src.model.ensemble_v2 import EnsembleV2
from src.model.model import Model
from src.space_predictor.space_predictor import space_predictor
from src.space_predictor.Iterative_space_predictor import Iterative_space_predictor

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.stub_model import StubModel


class MRR_CHARS(HitAtK):
    def calculate(self, model: Model, data: str or List[dict],char_weight:float=None, space_predictor: space_predictor=Iterative_space_predictor) -> float:
        """
        ** this MRR metric for chars consider the rank of the prediction and divide it by 1.
        For example:
        model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 0.75), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפי', 1)]),]))
        data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
        Because we look on the chars, the right prediction א rank on second place (the prediction sorted by their score) so it will get 0.5.
        The chars ו,ה,ב rank on the first prediction so each char will get 1
        For עוגות the right prediction ו rank on first place so it will get 1.
        We had 5 cases of masking ( 5 chars to predict).
        As a result we except to get the result: (0.5+1+1+1+1)/5 = 0.9.

        :param model: model to be tested
        :param data: data to be tested on. could be string if its a path to test json file or dict if its a ready to go
                     list in form of [{"text": "...", "missing": {...}}]
        :return: MRR score (As explained above)
        """

        if isinstance(data, str):
            data = self.get_data_at_hit_at_k_test_format(data)
        results = []
        progress_bar = tqdm(range(len(data)), desc=f"{model.model_path} Char MRR", unit="entry",
                            bar_format="\033[32m{l_bar}{bar}{r_bar}\033[0m")
        for entry_idx, entry in enumerate(data):
            progress_bar.update(1)
            current_text=entry['text']
            real_values = list(entry['missing'].values())
            if(real_values==[]):
                continue
            # if(space_predictor):
            #
            if char_weight is not None and isinstance(model, EnsembleV2):
                modelRes = model.predict(current_text,char_model_weight=char_weight).get_prediction_sorted()
            else:
                modelRes = model.predict(current_text).get_prediction_sorted()
            list_of_preds = self._model_result_to_list_of_preds(modelRes)
            after_sp_text = model.text
            miss, hit = self._comp_after_sp_texts(after_sp_text, entry['missing'])
            real_values = list(miss.values())
            char_lst = []
            for pred_idx, preds in enumerate(list_of_preds):
                missing_idxs = self._get_missing_idxs(pred_idx, after_sp_text)

                for j in missing_idxs:
                    pred_missing_chars = []
                    for w in preds:
                        pred_missing_chars.append(w[j])
                    char_lst.append(pred_missing_chars)

            fit_count = hit
            for real_val_idx, real_val in enumerate(real_values):
                if len(char_lst) != 0:
                    if real_val in char_lst[real_val_idx]:
                        index_rank = char_lst[real_val_idx].index(real_val)+1
                        fit_count += 1/index_rank
            if len(real_values):
                results.append(fit_count / len(real_values))
            else:
                results.append(0)
        return sum(results) / len(results)

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
                    return [c_i for c_i, c in enumerate(word) if c=='?']


