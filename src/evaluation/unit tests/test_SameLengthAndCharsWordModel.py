from typing import List

from src.classes.prediction import Prediction
from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel
import re
import config
from src.model.standard_model import StandardModel


def is_correct_known_chars(text, pred_text):
    for i, c in enumerate(text):
        if c == '?':
            continue
        else:
            assert c == text[i]


model = SameLengthAndCharsWordModel(config.configs['word_model_path'])
text = 'שלום לכם ילדי? וילדות ??? י??ל המבולבל'


def test_order_is_correct():
    res = model.predict(text)
    only_knowns = [textpart.text for textpart in res if textpart.text != '?']
    assert only_knowns == ['שלום', 'לכם', 'וילדות', 'המבולבל']


def test_all_words_same_length():
    res = model.predict(text)
    only_missing = res.get_only_predictions()
    for textpart in only_missing:
        is_all_words_same_len = all(
            len(pred.value) == len(textpart.predictions[0].value) for pred in textpart.predictions)
        assert is_all_words_same_len


def test_has_correct_known_chars():
    res = model.predict(text)
    only_missing = res.get_only_predictions()
    expected_substrings = ['ילדי?', '???', 'י??ל']
    for i in range(len(only_missing)):
        txt_preds = [pred.value for pred in only_missing[i].predictions]
        for txt_pred in txt_preds:
            is_correct_known_chars(expected_substrings[i], txt_pred)


def test_all_preds_has_correct_known_chars():
    res = model.predict(text)
    text_lst = text.split()
    dic = {}
    filtered_words = [word for word in text_lst if any(c.isalpha() for c in word) and '?' in word]
    for i in range(len(text_lst)):
        if text_lst[i] in filtered_words:
            dic[i] = text_lst[i]
    for i in range(len(res)):
        if i in dic:
            assert validate_exsisting_chars([res[i].predictions],dic.get(i))


def validate_exsisting_chars(presd_lst:list,word:str)->bool:
    values = get_list_of_str_preds(presd_lst)
    pattern = word.replace("?", "\\w")
    filtered_words = [w for w in values[0] if re.search(pattern, w)]
    return filtered_words==values[0]


def get_list_of_str_preds(predictions: List[Prediction]) -> List[List[str]]:
    list_of_preds = []
    for l in predictions:
        if l == None:
            continue
        preds = []
        for pred in l:
            preds.append(pred.value)
        list_of_preds.append(preds)
    return list_of_preds