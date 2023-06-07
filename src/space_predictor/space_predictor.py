from abc import ABC

from src.model.model import Model


class space_predictor(ABC):
    def genText(self,text:str,threshold:int)->str:
        pass