
from src.model.model import Model
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
        indeces_of_spaces=[]
        char_model=StandardModel(config.configs['char_model_path'])
        model_predictions=char_model.predict(text)
        for text_part in model_predictions:
            if text[index_in_text]==' ':
                index_in_text+=1
            if text_part.predictions== None:
                index_in_text+=len(text_part.text)
            else:
                for prediction in text_part.predictions:
                    if prediction.value=='':
                        indeces_of_spaces.append(index_in_text)
                index_in_text+=1
        new_text=self.replace_with_spaces(text,indeces_of_spaces)
        return new_text

    def replace_with_spaces(self, text, indices):
        # Convert the text string to a list of characters
        text_list = list(text)

        # Iterate over the indices list
        for index in indices:
            # Check if the index is within the valid range
            if index >= 0 and index < len(text_list):
                # Replace the character at the given index with a space
                text_list[index] = ' '

        # Convert the list of characters back to a string
        replaced_text = ''.join(text_list)

        return replaced_text