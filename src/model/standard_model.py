import torch

from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.model.model import Model
from src.utils.strings import StringUtils

class StandardModel(Model):
    def predict(self,text:str, min_p : float =0.01):
        if '?' not in text:
            return [TextPart(text,None)]
        # predict
        preds = self._get_predictions(text)

        # split to parts so we can turn these parts to list of {'text': 'part', 'predictions': [...]}
        splited_text=StringUtils.split_text(text)
        #
        res=[]
        pred_index=0
        for i,part in enumerate(splited_text):
            next_pred=TextPart(part,None)
            if '?' in part:
                next_pred.predictions= list(filter(lambda x : x.score>=min_p , preds[pred_index]))
                pred_index+=1
            res.append(next_pred)

        return res

    def _get_predictions(self,text):
        text=StringUtils.insert_masks(text)
        # tokenize input
        token_ids = self.tokenizer.encode(text, return_tensors='pt')
        # masked token indexes
        masked_pos = [mask.item() for mask in (token_ids.squeeze() == self.tokenizer.mask_token_id).nonzero()]
        # predict
        with torch.no_grad():
            output = self.model(token_ids)
        last_hidden_state = output[0].squeeze()
        # extract predictions from last_hidden_state
        res = []
        for mask_index in masked_pos:
            mask_hidden_state = last_hidden_state[mask_index]
            topk_predictions=torch.topk(mask_hidden_state, k=100, dim=0)
            topk_preds_idx = topk_predictions[1]
            predictions = [self.tokenizer.decode(i.item()).strip() for i in topk_preds_idx]
            predictions=[i.replace('##', '') for i in predictions]
            probs = topk_predictions[0].softmax(-1)
            # parse the data into interface
            cur_pred_results = []
            for pred_index, pred in enumerate(predictions):
                cur_pred_results.append(Prediction(pred,round(probs[pred_index].item(), 2)))
            res.append(cur_pred_results)

        return res



