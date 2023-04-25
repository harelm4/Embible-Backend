import itertools

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.standard_model import StandardModel


class CharModel(StandardModel):

    def predict(self, text: str, min_p: float = 0.01) -> ModelResult:
        """
        main function of this class, genetates predictions
        :param text: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param min_p: minimal value of prediction score
        :return: charModelResult , an object encapsulating list of TextParts
        """
        if '?' not in text:
            return ModelResult([TextPart(text, None)])
        preds_w = super(CharModel, self).predict(text)
        pres_only = preds_w.get_only_predictions()
        predictions = [x.predictions for x in preds_w]
        list_of_preds = []
        for l in predictions:
            if l == None:
                continue
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)
        list_of_p = []
        for i in range(len(list_of_preds)):
            p_lst = []
            for j in range(len(list_of_preds[i])):
                p_lst.append(pres_only[i].predictions[j].score)
            list_of_p.append(p_lst)
        splitted_sent = text.split()
        pred_index = 0
        res = []
        for word in range(len(splitted_sent)):
            next_pred = TextPart(splitted_sent[word], None)
            if '?' in splitted_sent[word]:
                next_pred.text = '?'
                num_of_q_marks = splitted_sent[word].count('?')
                completions, scores = self.fill_word_preds(splitted_sent[word],
                                                           list_of_preds[pred_index:pred_index + num_of_q_marks],
                                                           list_of_p[pred_index:pred_index + num_of_q_marks])
                next_pred.predictions = list(filter(lambda x: x.score >= min_p, self.merge_preds(completions, scores)))
                pred_index += num_of_q_marks
            res.append(next_pred)
        return ModelResult(res)

    def merge_preds(self, preds: list, p_lst: list) -> list:
        """
        this function merges the predictions and the scores into a list of prediction class
        :param p_lst: List of float scores .example [0.55,0.41,0.99,0.1]
        :param preds: list of lists of predictions .example: [['א',ב','כ,'ר']]
        :return: list of prediction class
        """
        merged_lst = []
        for pred, score in zip(preds, p_lst):
            merged_lst.append(Prediction(pred, score))
        return merged_lst

    def fill_word_preds(self, text: str, preds: list, p_lst: list) -> list:
        """
        this function fills in all the combinations of the predictions inside the text given
        :param text: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param preds: list of lists of predictions .example: [[א,ב],[כ,ר]]
        :return: list of all the combinations of the predictions filled in the text
        """
        placeholders = [i for i in range(len(text)) if text[i] == "?"]
        num_placeholders = len(placeholders)
        char_lists = [lst for lst in preds if lst]  # remove empty lists
        if not char_lists or num_placeholders != len(char_lists):
            return []

        completions = []
        scores = []
        for chars, score in zip(itertools.product(*char_lists), itertools.product(*p_lst)):
            if len(chars) != num_placeholders:
                continue
            completion = list(text)
            for i, c in zip(placeholders, chars):
                completion[i] = c
            if len(''.join(completion)) != len(text):
                continue
            completions.append(''.join(completion))
            scores.append(sum(score) / num_placeholders)
        return completions, scores
