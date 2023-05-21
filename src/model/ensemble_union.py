from typing import List

import config
from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel
from src.model.ensemble_v2 import EnsembleV2

from src.model.model import Model
from src.model.Char_model import CharModel

import config
from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel

from src.model.model import Model
from src.model.Char_model import CharModel


class EnsembleUnion(EnsembleV2):
    def __init__(self):
        self.model_path = 'EnsembleUnion'
        self.char_model = CharModel(config.configs['char_model_path'])
        self.word_model = SameLengthAndCharsWordModel(config.configs['word_model_path'])

    def merge_textparts(self, char_textpart: TextPart, word_textpart: TextPart, char_model_weight: float) -> TextPart:
        """
        Merges  `char_textpart` and `word_textpart` into one TextPart that all the predictions are apear in both previous
        texparts. the formula of the score is char_model_weight * char_pred.score + (1-char_model_weight) * word_pred.score
        :param char_textpart: A TextPart of the char model
        :param word_textpart: A TextPart of the word model
        :param char_model_weight: the weight being used in the formula
        :return: A merged TextPart
        """
        new_scores=self.Weighted_average(char_textpart.predictions, word_textpart.predictions, char_model_weight, 1-char_model_weight)
        res=[Prediction(value,score) for value,score in new_scores.items()]
        return TextPart('?', list(sorted(res, key=lambda x: x.score, reverse=True)))

    def Weighted_average(self, list1, list2, weight1, weight2):

        new_scores = {}

        # Create a dictionary to store scores from list1
        scores1 = {obj.value: obj.score for obj in list1}

        # Create a dictionary to store scores from list2
        scores2 = {obj.value: obj.score for obj in list2}

        # Iterate over the keys in scores1
        for value in scores1.keys():
            # Initialize score to 0 if the value is not in scores2
            if value not in scores2:
                score2 = 0
            else:
                score2 = scores2[value]

            # Calculate the new score using the given formula
            new_score = weight1 * scores1[value] + weight2 * score2

            # Store the new score in the dictionary
            new_scores[value] = new_score

        # Iterate over the keys in scores2
        for value in scores2.keys():
            # Skip values that are already processed
            if value in new_scores:
                continue

            # Initialize score to 0 if the value is not in scores1
            if value not in scores1:
                score1 = 0
            else:
                score1 = scores1[value]

            # Calculate the new score using the given formula
            new_score = weight1 * score1 + weight2 * scores2[value]

            # Store the new score in the dictionary
            new_scores[value] = new_score

        return new_scores
