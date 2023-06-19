from abc import ABC
from typing import List

from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.classes.model_result import ModelResult
from src.classes.text_part import TextPart


class Model(ABC):
    def __init__(self,model_path,mask_token='[MASK]'):
        self.mask_token=mask_token
        self.model_path=model_path
        self.model=AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)
        self._topk_predictions_raw=100

    def predict(self,text:str,min_p:float)->ModelResult:
        """
        the main function of the model class.
        call this function to get prediction of a text with a missing parts.
        the missing parts anotation is `?`.
        :param text: the text to predict
        :param min_p: min value of the score
        """
        pass