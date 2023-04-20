from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.model import Model


class StubModel(Model):

    def __init__(self,modelRes:ModelResult):
        """
        this is a stub model for testing
        :param modelRes: a model result that when calling predict, the user gets exactly that
        """
        self.modelRes=modelRes
        pass

    def predict(self,text):
        return self.modelRes
