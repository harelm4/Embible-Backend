import config
from src.model.model import Model
from src.model.standard_model import StandardModel
from src.space_predictor.space_predictor import space_predictor
from src.model.standard_model import StandardModel
import config
import copy
class Iterative_space_predictor(space_predictor):
    def genText(self,text:str,threshold=0.5)->(str):
        """
        In this function we will predict al of the spaces in the text and put them in the text above certain threshold of certainty
        :param threshold: certain threshold of certainty
        :param model: model's predictions
        :param text:input text full of ? that might be spaces
        :return text with certain spaces
        """

        index_in_text=0
        time_of_spaces=0
        model_res=StandardModel(config.configs['char_model_path']).predict(text)
        for index,text_part in enumerate(model_res.lst):
            if(text_part.text!='?'):
                index_in_text+=len(text_part.text)
            else:
                for prediction in text_part.predictions:
                    if(prediction.score>threshold and prediction.value==""):
                        text=text[:index_in_text] + " " + text[index_in_text+1:]
                        index_in_text+=1
                        model_res.lst.pop(index-time_of_spaces)
                        time_of_spaces+=1
        return text
