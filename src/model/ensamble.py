from typing import List

import config
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.classes.ensemble_result import EnsembleResult
from src.model.model import Model
from src.model.sequential_model import SequentialModel
from src.model.standard_model import StandardModel
from src.utils.strings import StringUtils


class Ensamble(Model):

    def __init__(self):
        self.word_model=StandardModel(config.configs['word_model_path'])
        self.sequential_char_model = SequentialModel(config.configs['char_model_path'])


    def predict(self,text:str,min_p=0.1)->EnsembleResult:
        """
        main function of this class, genetates predictions
        :param text: text from the user .example :  "ויב?א ה את הש??ם ואת ה?רץ"
        :param min_p: minimal value of prediction score
        :return: EnsembleResult , an object encapsulating list of TextParts
        """
        self.last_word_model_preds=self.word_model.predict(text,min_p)
        self.last_char_model_sequential_preds = self.sequential_char_model.predict(text,min_p)

        self.splited_text = StringUtils.split_text(text)
        res=[]
        self.pred_index=0
        for i in range(len(self.splited_text)):
            if self.pred_index>=len(self.splited_text):
                break
            if self.splited_text[self.pred_index]!='?':
                res.append(self.last_word_model_preds[self.pred_index])
            else:
                res.append(self._get_pred(self.pred_index))
            self.pred_index+=1


        return EnsembleResult(list(filter(lambda x:x!=None,res)))


    def _get_pred(self,pred_index:int):
        if(len(self.last_word_model_preds)<=pred_index):
            return
        # handle extreme points
        if '?' not in self.last_word_model_preds[pred_index].text:
            return self.last_word_model_preds[pred_index]

        before,after=self._get_before_after(pred_index)
        return self._get_pred_by_type( before, after, pred_index)

    def _get_before_after(self,pred_index:int):
        if pred_index ==0:
            before=TextPart(' ',None)
        else:
            before=self.last_word_model_preds[pred_index-1]
        if pred_index ==len(self.last_word_model_preds)-1:
            after = TextPart(' ',None)
        else:
            after = self.last_word_model_preds[pred_index + 1]

        return before,after

    def _get_pred_by_type(self,before:TextPart,after:TextPart,pred_index:int)->TextPart:
        if self._is_index_starts_a_word(pred_index):
            return self._get_word_prediction_at_index(pred_index)
        else:
            return self._get_char_or_subword_prediction_at_index(pred_index)

    def _get_word_prediction_at_index(self,index:int)->TextPart:
        preds=self.last_word_model_preds[index].predictions
        res_preds=[]
        word_len=self._get_this_word_length(index,self.last_word_model_preds)
        for pred in preds:
            if len(pred.value)==word_len:
                res_preds.append(pred)
        if len(res_preds)==0:
            # in case non of the predictions are the length of the word
            res_preds=self._get_pred_sequentially_at_index(index, word_len)
        return TextPart('?',res_preds)

    def _get_char_or_subword_prediction_at_index(self,pred_index):
        subword_len=0
        x=len(self.splited_text[pred_index:])
        for c in self.splited_text[pred_index:]:
            if c != '?':
                break
            subword_len += 1
        res_preds=self._get_pred_sequentially_at_index(pred_index, subword_len)
        return TextPart('?', res_preds)


    def _get_this_word_length(self,index,preds):
        count=0
        for i in range(index,len(preds)):
            if(preds[i].text=='?'):
                count+=1;
            else:
                return count
        return count

    def _get_pred_sequentially_at_index(self, index:int, word_len:int)->List[Prediction]:
        preds=[pred.predictions[0] for pred in self.last_char_model_sequential_preds[index:index+word_len] 
               if pred.predictions !=None and pred.predictions!=[] ]
        if preds==[]:
            return []
        res_txt=''
        pred_score_sum=0
        for pred in preds:
            res_txt+=pred.value
            pred_score_sum+=pred.score
        avg_score=pred_score_sum/len(preds)
        self.pred_index += word_len - 1
        return [Prediction(res_txt,round(avg_score,3))]

    def _is_index_starts_a_word(self,pred_index:int)->bool:
        for i in range(pred_index,len(self.splited_text)):
            if self.splited_text[i]==' ':
                break
            if self.splited_text[i]!='?':
                return False
        return True

