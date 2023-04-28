from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.model import Model
from src.model.standard_model import StandardModel
from src.model.word_model import wordModel
from src.utils.strings import StringUtils



class SameLengthWordModel(wordModel):

    def predict(self,text:str,min_p:float):
        modelResult=super.predict(text,min_p)
        return self.filter_predictions_by_size(modelResult,text)


    def filter_predictions_by_size(self,modelResult:ModelResult,text:str):
        """
        gets a model result and filters all the results which are not in the wanted size
        :returns modelResult with filtered outputs
        :param modelRes: a model result that when calling predict, the user gets exactly that
        :param text the input text by the user
        """
        len_of_masks = StringUtils.index_of_words_contains_question_mark(text)
        masks_index=0
        for i,text_part in enumerate(modelResult.lst):
            if(text_part.predictions!=None):
                new_preds=[]
                for pred in text_part.predictions:
                    if(len(pred.value)==len_of_masks[masks_index]):
                        new_preds.append(pred)
                masks_index+=1
                text_part.predictions=new_preds
                modelResult.lst[i]=text_part
        return modelResult

