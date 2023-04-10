#
# from src.model.model import Model
# from src.space_predictor.space_predictor import space_predictor
#
# class Iterative_space_predictor(space_predictor):
#     def genText(self,model:Model,text:str,threshold=0.1)->(str,Model):
#         """
#         In this function we will predict spaces in the text untill
#         :param threshold: certain threshold of certainty
#         :param model: model's predictions
#         :param text:input text full of ? that might be spaces
#         :return text with certain spaces
#         """
#
#
from huggingface_hub import login

import config
from src.model.standard_model import StandardModel
from src.space_predictor.Iterative_space_predictor import Iterative_space_predictor
login(config.configs['hf_token'])
r= Iterative_space_predictor().genText(StandardModel(f'Embible/mBert-20-epochs'),'???שש?')
print(r)