import itertools

from src.classes.model_result import ModelResult
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
        sm = StandardModel('Embible/tavbert-50-epochs')
        if '?' not in text:
            return ModelResult([TextPart(text, None)])
        preds = sm.predict(text, min_p)
        predictions = [x.predictions for x in preds]
        list_of_preds = []
        for l in predictions:
            if l == None:
                continue
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)
        return ModelResult(list_of_preds)

    def fill_word_preds(self, text: str, preds: list) -> list:
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
        for chars in itertools.product(*char_lists):
            if len(chars) != num_placeholders:
                continue
            completion = list(text)
            for i, c in zip(placeholders, chars):
                completion[i] = c
            completions.append(''.join(completion))
        return completions

    def get_predictions_dict(self, text: str, full_pred: list) -> dict:
        """
        this function generates a dictionary of words with sub masked as key and all the possible word predictions as
        values
        :param text: text from the user .example :  "אני או?ב שו??לד"
        :param full_pred: list of all the combinations of the predictions filled in the text
        :return: dictionary of words with sub masked as key and all the possible word predictions as
        values .example : {'או?ב': ['אויב', 'אוהב', 'אוכב'], 'שו??לד': ['שובולד','שויתלד', 'שוקולד', 'שויבלד']}
        """
        splitted_sent = text.split()
        q_mark_indexes = []
        pred_dict = {}
        for word in range(len(splitted_sent)):
            if '?' in splitted_sent[word]:
                q_mark_indexes.append(word)
                pred_dict[splitted_sent[word]] = []
        for i in q_mark_indexes:
            for pred in full_pred:
                if pred.split()[i] not in pred_dict[splitted_sent[i]]:
                    pred_dict[splitted_sent[i]] += [pred.split()[i]]
        return pred_dict
