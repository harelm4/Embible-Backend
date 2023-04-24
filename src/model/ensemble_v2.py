from typing import List

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.model import Model
# from src.model.CharModel import CharModel
# from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel
from src.model.stub_model import StubModel
from src.utils.strings import StringUtils


class EnsembleV2(Model):
    def __init__(self):
        self.model_path = 'EnsembleV2'
        # text = "אני ???? שוקולד וע?גות גבינה"
        # stubs , TODO - replace to real models when ready
        self.char_model = StubModel(ModelResult([
            TextPart('אני ', None),
            TextPart('?', [Prediction('אוהב', 0.4), Prediction('שותה', 0.5)]),
            TextPart(' שוקולד וע', None),
            TextPart('?', [Prediction('ועוגות', 1), Prediction('וערגות', 1)]),
            TextPart('גות גבינה', None),
        ]))


        self.word_model = StubModel(ModelResult([
            TextPart('אני ', None),
            TextPart('?', [Prediction('אוהב', 1), Prediction('שונא', 0.1)]),
            TextPart(' שוקולד וע', None),
            TextPart('?', [Prediction('ועוגות', 0.8), Prediction('וחוגות', 0.1)]),
            TextPart('גות גבינה', None),
        ]))

    def predict(self, text: str, min_p: float = 0.1, char_model_weight: float = 0.5,
                space_predictor=None) -> ModelResult:
        """
        predicting based on word model and char model
        :param text: the text to predict , for example "אני ???? שוקולד וע?גות גבינה"
        :param min_p: minimum score (probability) of each prediction
        :param char_model_weight: the weight of the char model result , it is used in the formula :
                char_model_weight * char_pred.score + (1-char_model_weight) * word_pred.score
                that is used for each conjoined prediction
        :param space_predictor: a SpacePredictor object that determines the strategy of predicting the spaces
        :return: ModelResult that includes the prediction after the activation of formula
                char_model_weight * char_pred.score + (1-char_model_weight) * word_pred.score
        """
        char_model_result = self.char_model.predict(text)
        if space_predictor:
            text, char_model_result = space_predictor.genText(char_model_result, text)
        word_model_result = self.word_model.predict(text)
        if len(char_model_result) != len(word_model_result):
            raise Exception(
                'there is something wrong with one of the models , the length of the predictions is not equal',
                char_model_result, word_model_result)

        return self.ensemble_predictions(char_model_result, word_model_result, char_model_weight)

    def ensemble_predictions(self, char_model_result: ModelResult, word_model_result: ModelResult,
                             char_model_weight: float) -> ModelResult:
        """
        Distinguish between a textual TextPart and a missing part TextPart, sends the missing part TextParts to a function
        that merges the word model one and the char model one.
        :param char_model_result: the ModelResult of the char model
        :param word_model_result: the ModelResult of the word model
        :param char_model_weight: the weight of the char model result , it is used in the formula :
                                    char_model_weight * char_pred.score + (1-char_model_weight) * word_pred.score
                                    that is used for each conjoined prediction
        :return: the end result of the ensemble
        """
        num_of_textparts = len(char_model_result)
        res = []
        for i in range(num_of_textparts):
            if char_model_result[i].text != '?':
                res.append(char_model_result[i])
            else:
                res.append(self.merge_textparts(char_model_result[i], word_model_result[i], char_model_weight))

        return ModelResult(res)

    def merge_textparts(self, char_textpart: TextPart, word_textpart: TextPart, char_model_weight: float) -> TextPart:
        """
        Merges  `char_textpart` and `word_textpart` into one TextPart that all the predictions are apear in both previous
        texparts. the formula of the score is char_model_weight * char_pred.score + (1-char_model_weight) * word_pred.score
        :param char_textpart: A TextPart of the char model
        :param word_textpart: A TextPart of the word model
        :param char_model_weight: the weight being used in the formula
        :return: A merged TextPart
        """
        res_preds = []
        for char_pred in char_textpart.predictions:
            for word_pred in word_textpart.predictions:
                if char_pred.value == word_pred.value:
                    score = char_model_weight * char_pred.score + (1 - char_model_weight) * word_pred.score
                    res_preds.append(Prediction(char_pred.value, score))

        return TextPart('?', list(sorted(res_preds, key=lambda x: x.score, reverse=True)))
