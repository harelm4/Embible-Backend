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
        input_text=text
        model_res=StandardModel(config.configs['char_model_path']).predict(text,threshold).get_only_predictions()
        index_in_model_res = 0
        for index, char in enumerate(text):
            if (char == "?"):
                text_part=model_res.lst[index_in_model_res]
                if (len(text_part.predictions) > 0):
                    for prediction in text_part.predictions:
                        if (prediction.score > threshold and prediction.value == ""):
                            text = text[:index] + " " + text[index + 1:]
                index_in_model_res+=1
        return text

