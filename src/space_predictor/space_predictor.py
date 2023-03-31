from abc import ABC

from src.model.model import Model


class space_predictor(ABC):
    def genText(self,model:Model,text:str)->str:
        pass