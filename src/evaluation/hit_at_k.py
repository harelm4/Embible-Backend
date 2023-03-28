
import sys
import pandas as pd
from src.model.model import Model



def calculate_hit_at_k(k: int, model: Model,data:dict,max_records=sys.maxsize,print_progress=True) -> float:
    """
    calculating hit at k for a model based on test set
    :param k: param of hit at k
    :param model: model to evaluate
    :return: hit@k score
    """

    hit_at_ks = []
    to_print=''
    for entry_idx,entry in enumerate(data):
        real_values = entry['missing'].values()
        predictions = model.predict(entry['text']).get_only_k_predictions(k).lst  # list of text parts
        predictions = [x.predictions for x in predictions]  # list of lists of predicion objects
        list_of_preds = []
        for l in predictions:
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)

        # === print loading state ===
        if print_progress:
            for c in to_print:
                sys.stdout.write('\b')
            to_print=f'{entry_idx+1}/{len(data)}'
            sys.stdout.write(to_print)
            sys.stdout.flush()
        #====
        hit_at_ks.append(_hit_at_k(list_of_preds, [x for x in real_values if x != '']))
        if entry_idx==max_records:
            break
    sys.stdout.write('\n')
    return (sum(hit_at_ks) / len(hit_at_ks))



def _hit_at_k(predictions, real_values):
    """
    private function to calculate hit@k
    real_values-list of words/characters
    predictions-list of lists while each list contains k words/characters
    example:
    k=2
    real_values=[שלום,ישראל]
    predictions=[[ישראל,יעקב],[חלום,ביטחון]]
    return-> 0.5
    """
    if all(len(x) == 1 for x in real_values) and len(predictions)!=len(real_values):
        new_preds=[]
        for pred_lst in predictions:
            index=0
            for c in pred_lst[0]:
                new_preds+=[x[index] for x in pred_lst]
                index+=1
        predictions=new_preds

    if len(predictions)==0:
        return 0
    count_mone, count_mechane = 0, 0
    try:
        for i, word in enumerate(real_values):
            if word in predictions[i]:
                count_mone += 1
            count_mechane += 1
        if count_mechane == 0:
            return 0
    except:
        print('hit@k loop error!!!')
        return 0

    return count_mone / count_mechane

def get_data_at_hit_at_k_test_format(file_path:str):
    df = pd.read_json(file_path, orient='records', lines=True)
    data = df.to_dict(orient='records')
    return [{'text':x['verse'],'missing':x['missing_dictionary']} for x in data]