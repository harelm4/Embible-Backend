from src.classes.model_result import ModelResult
from src.classes.text_part import TextPart
from src.model.standard_model import StandardModel


class wordModel(StandardModel):

    def predict(self, text: str, min_p: float = 0.01) -> ModelResult:
        """
        main function of this class, genetates predictions
        :param text: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param min_p: minimal value of prediction score
        :return: wordModelResult , an object encapsulating list of TextParts
        """
        if '?' not in text:
            return ModelResult([TextPart(text, None)])
        text_lst = [word if '?' not in word else '?' for word in text.split()]
        preds_w = super(wordModel, self).predict(" ".join(text_lst))
        pres_only = preds_w.get_only_predictions()
        res = []
        for dict in preds_w:
            if '?' not in dict.text:
                splitted_sent = dict.text.split()
                for word in range(len(splitted_sent)):
                    next_pred = TextPart(splitted_sent[word], None)
                    res.append(next_pred)
            else:
                res.append(dict)
        return ModelResult(res)
