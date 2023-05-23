from src.classes.model_result import ModelResult
from src.model.model import Model
from src.space_predictor.space_predictor import space_predictor
from src.space_predictor.Iterative_space_predictor import Iterative_space_predictor

class Recurrent_space_predictor(space_predictor):
    def genText(self,text:str,threshold=0.1)->str:
        """
        In this function we will predict spaces in the text untill
        :param threshold: certain threshold of certainty
        :param model: model's predictions
        :param text:input text full of ? that might be spaces
        :return text with certain spaces
        """

        new_txt= Iterative_space_predictor().genText(text,threshold)
        if text==new_txt:
            return new_txt
        else:
            return self.genText(new_txt,threshold)
