
import sys
import pandas as pd
from src.model.model import Model
from src.utils.strings import StringUtils


def calculate_hit_at_k(k: int, model: Model,data:dict,max_records=sys.maxsize,print_progress=True) -> float:
    """
    calculating hit at k for a model based on test set
    :param k: param of hit at k
    :param model: model to evaluate
    :return: hit@k score
    """

    hit_at_ks = []
    to_print=''
    total_full_words=[]
    for entry_idx,entry in enumerate(data):
        real_values = entry['missing'].values()
        total_full_words+=StringUtils.find_question_word_index(entry)
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
    # print(total_full_words)
    return (sum(hit_at_ks) / len(hit_at_ks))


def preprocess_text_for_words(entry):
    words = entry.split()
    output_words=[]
    for i,word in enumerate(words):
        if all(c == '?' for c in word):
            output_words.append('?')
        else:
            output_words.append(word)

    return ' '.join(output_words)


def get_words_from_missing(missing_dict, list_of_indeces):
    total_words={}
    for i,tup in enumerate(list_of_indeces):
        s=""
        for j in range(tup[1]):
            key=j+tup[0]
            key=str(key)
            s+=missing_dict[key]
        total_words[i]=s
    return total_words


def get_predictions_for_words(predictions, word_indeces):
    missing=[]
    for x in predictions.lst:
        if(x.predictions != None):
            missing.append(x)
    res=[]
    for word_index in word_indeces:
        res.append(missing[word_index])
    return res


def get_word_indeces(txt):
    counter_of_qm=0
    indeces=[]
    for i,c in enumerate(txt):
        if(c=="?"):
            if(i>0 and i<len(txt)-1):
                if(txt[i-1]==" " and txt[i+1]==" "):
                    indeces.append(counter_of_qm)
            elif(i==0 and txt[1]==" "):
                    indeces.append(counter_of_qm)
            elif(i==len(txt)-1 and txt[i-1]==" "):
                    indeces.append(counter_of_qm)
            counter_of_qm+=1
    return indeces


def calculate_word_hit_at_k(k: int, model: Model,data:dict,max_records=sys.maxsize,only_words=False) -> float:
    """
    calculating hit at k for a model based on test set
    :param k: param of hit at k
    :param model: model to evaluate
    :return: hit@k score
    """

    hit_at_ks = []
    total_full_words=[]
    for entry_idx,entry in enumerate(data):
        if(len(StringUtils.find_question_word_index(entry))==0):
            continue
        #lens of all masks of full words by their order
        len_of_masks = [t[1] for t in StringUtils.find_question_word_index(entry)]
        #dictionary of {index of masked word:word}
        if(only_words):
            word_dict = dict([(i, value) for i, value in enumerate(entry['missing'].values())])
        else:
            word_dict=get_words_from_missing(entry['missing'],StringUtils.find_question_word_index(entry))
        #input text
        txt=entry['text']
        #input text after transform each sequence of ? to single ?
        txt=preprocess_text_for_words(txt)
        words_indeces=get_word_indeces(txt)
        predictions = model.predict(txt)  # list of text parts
        predictions_for_words = get_predictions_for_words(predictions,words_indeces)
        list_of_k_words_per_prediction=[]
        for index,list_of_predictions in enumerate(predictions_for_words):
            len_of_mask=len_of_masks[index]
            l=[pred.value for pred in list_of_predictions.predictions if len(pred.value)==len_of_mask]
            list_of_k_words_per_prediction.append(l[:k])
        hit_at_ks.append(_hit_at_k( list_of_k_words_per_prediction,word_dict.values()))
        if entry_idx==max_records:
            break
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
        print(real_values)
        print(predictions)
        print('hit@k loop error!!!')
        return 0

    return count_mone / count_mechane

def get_data_at_hit_at_k_test_format(file_path:str):
    df = pd.read_json(file_path, orient='records', lines=True)
    data = df.to_dict(orient='records')
    return [{'text':x['verse'],'missing':x['missing_dictionary']} for x in data]