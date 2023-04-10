
from src.model.model import Model
from src.space_predictor.space_predictor import space_predictor
import copy
class Iterative_space_predictor(space_predictor):
    def genText(self,model:Model,text:str,threshold=0.1)->(str,Model):
        """
        In this function we will predict al of the spaces in the text and put them in the text above certain threshold of certainty
        :param threshold: certain threshold of certainty
        :param model: model's predictions
        :param text:input text full of ? that might be spaces
        :return text with certain spaces
        """

        index_in_text=0
        time_of_spaces=0
        new_model_lst=copy.deepcopy(model.lst)
        for index,dict_of_predictions in enumerate(model.lst):
            if(dict_of_predictions.text!='?'):
                index_in_text+=len(dict_of_predictions.text)
            else:
                for prediction in dict_of_predictions.predictions:
                    if(prediction.score>threshold and prediction.value==""):
                        text=text[:index_in_text] + " " + text[index_in_text+1:]
                        index_in_text+=1
                        new_model_lst.pop(index-time_of_spaces)
                        time_of_spaces+=1


        model.lst=new_model_lst
        return text,model