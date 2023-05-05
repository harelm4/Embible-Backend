from typing import List, Tuple

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.SameLengthWordModel import SameLengthWordModel

from src.model.model import Model
from src.model.standard_model import StandardModel
from src.utils.strings import StringUtils


class SameLengthAndCharsWordModel(SameLengthWordModel):

    def predict(self, text: str, min_p: float = 0.01):
        modelResult = super(SameLengthAndCharsWordModel,self).predict(text, min_p)
        return self.filter_predictions_by_chars(modelResult, text)

    def filter_predictions_by_chars(self, modelResult: ModelResult, text: str):
        """
        gets a model result and filters all the results which are not in the wanted size
        :returns modelResult with filtered outputs
        :param modelRes: a model result that when calling predict, the user gets exactly that
        :param text the input text by the user
        """
        # {index_of_masked_word:(index of char in word,char)}
        known_cars_dict = StringUtils.get_known_chars_and_their_indeces(text)
        masks_index = 0
        for i, text_part in enumerate(modelResult.lst):
            if (text_part.predictions != None):
                new_preds = []
                for pred in text_part.predictions:
                    flag = self.does_word_contains_known_chars(pred.value, known_cars_dict[i])
                    if (flag):
                        new_preds.append(pred)
                masks_index += 1
                text_part.predictions = new_preds
                modelResult.lst[i] = text_part
        return modelResult

    def does_word_contains_known_chars(self, pred_by_model: str, known_cars_lst: List[Tuple[int, str]]):
        """
        :param pred_by_model:word prediction by the model
        :param known_cars_lst: list of tuples of [(index of char in word,char)]
        :return: true if all known chars are in prediction else false
        """
        for i, c in known_cars_lst:
            if pred_by_model[i] != c:
                return False
        return True