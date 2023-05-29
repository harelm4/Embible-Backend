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
        ** this hit@k works only for same word length models **
        calculate hit@k score for chars.
        :param model: model to be tested
        :param data: data to be tested on. could be string if its a path to test json file or dict if its a ready to go
                     list in form of [{"text": "...", "missing": {...}}]
        :param k: the k of hit@k
        :return: hit@k score (# of char hists at k)/(# of missing chars)
        """
        if isinstance(data, str):
            data = self.get_data_at_hit_at_k_test_format(data)
        results=[]
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

            char_lst = []
            for pred_idx, preds in enumerate(list_of_preds):
                missing_idxs = self._get_missing_idxs(pred_idx, current_text)

                for j in missing_idxs:
                    pred_missing_chars = []
                    for w in preds:
                            pred_missing_chars.append(w[j])
                    char_lst.append(pred_missing_chars)


            fit_count = 0
            for real_val_idx, real_val in enumerate(real_values):
                if real_val in char_lst[real_val_idx]:
                    index_rank=char_lst[real_val_idx].index(real_val)+1
                    fit_count += 1/index_rank
            if len(real_values):
                results.append(fit_count / len(real_values))
            else:
                results.append(0)
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


chak = MRR_CHARS()
model = StubModel(ModelResult([
    TextPart('?', [Prediction('אוהב', 0.8), Prediction('עוהב', 1)]),
    TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפים', 1)]),
]))
data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
print(chak.calculate(model, data))
