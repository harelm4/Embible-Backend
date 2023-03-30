from abc import ABC

from src.model.model import Model


class spacePredictor(ABC):
    def genText(self,model:Model,text:str)->str:
        pass