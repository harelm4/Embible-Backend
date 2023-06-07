import itertools
from typing import List

from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.standard_model import StandardModel


class CharModel(StandardModel):

    def predict(self, text: str, min_p: float = 0.0001) -> ModelResult:
        """
        main function of this class, genetates predictions
        :param text: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param min_p: minimal value of prediction score
        :return: charModelResult , an object encapsulating list of TextParts
        """
        if '?' not in text:
            return ModelResult([TextPart(text, None)])
        preds_w = super(CharModel, self).predict(text, min_p)
        pres_only = preds_w.get_only_predictions()
        predictions = [x.predictions for x in preds_w]

        list_of_preds = self._get_list_of_str_preds(predictions)
        list_of_scores = self._get_list_of_scores(list_of_preds, pres_only)

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
                                                           list_of_scores[pred_index:pred_index + num_of_q_marks])
                next_pred.predictions = list(filter(lambda x: x.score >= min_p, self.merge_preds(completions, scores)))
                pred_index += num_of_q_marks
            res.append(next_pred)
        modelResult=ModelResult(res)
        for text_index,text_part in enumerate(modelResult.lst):
            if text_part.text=='?':
                modelResult[text_index].predictions=modelResult[text_index].predictions[:100]
        return modelResult

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

    def fill_word_preds(self, word: str, preds: list, p_lst: list) -> list:
        """
        this function fills in all the combinations of the predictions inside the text given
        :param word: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param preds: list of lists of predictions .example: [[א,ב],[כ,ר]]
        :return: list of all the combinations of the predictions filled in the text
        """
        placeholders = [i for i in range(len(word)) if word[i] == "?"]
        num_placeholders = len(placeholders)
        char_lists = [lst for lst in preds if lst]  # remove empty lists
        if not char_lists or num_placeholders != len(char_lists):
            return [],[]

        completions = []
        scores = []
        comb_chars=self.limit_product(char_lists,1000)#list(itertools.product(*char_lists))[:1000]
        comb_scores=self.limit_product(p_lst,1000)#list(itertools.product(*p_lst))[:1000]
        for chars, score in zip(comb_chars,comb_scores):
            if len(chars) != num_placeholders:
                continue
            completion = list(word)
            for i, c in zip(placeholders, chars):
                completion[i] = c
            if len(''.join(completion)) != len(word):
                continue
            completions.append(''.join(completion))
            scores.append(sum(score) / num_placeholders)
        return completions, scores

    def _get_list_of_str_preds(self, predictions: List[Prediction]) -> List[List[str]]:
        list_of_preds = []
        for l in predictions:
            if l == None:
                continue
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)
        return list_of_preds

    def _get_list_of_scores(self, list_of_preds, pres_only):
        list_of_p = []
        for i in range(len(list_of_preds)):
            p_lst = []
            for j in range(len(list_of_preds[i])):
                p_lst.append(pres_only[i].predictions[j].score)
            list_of_p.append(p_lst)
        return list_of_p

    def limit_product(self,char_lists, limit):
        # Calculate the total number of instances
        total_instances = 1
        for char_list in char_lists:
            total_instances *= len(char_list)

        # Determine the number of instances to slice
        num_instances = min(total_instances, limit)

        # Use itertools.islice to get the limited instances
        limited_product = itertools.islice(itertools.product(*char_lists), num_instances)

        return list(limited_product)