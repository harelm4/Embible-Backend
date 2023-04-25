from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel

import config
from src.model.standard_model import StandardModel

def is_correct_known_chars(text,pred_text):
    for i,c in pred_text:
        if c=='?':
            continue
        else:
            assert c==text[i]

model = SameLengthAndCharsWordModel(config.configs['word_model_path'])
text = 'שלום לכם ילדי? וילדות ??? י??ל המבולבל'
def test_order_is_correct():
    res = model.predict(text)
    only_knowns = [textpart.text for textpart in res if textpart.text!='?']
    assert only_knowns == ['שלום לכם ילדי',' וילדות ',' י','ל המבולבל']


def test_all_words_same_length():
    res = model.predict(text)
    only_missing = res.get_only_predictions()
    for textpart in only_missing:
        is_all_words_same_len = all(len(pred.value) == textpart.predictions[0].value  for pred in textpart.predictions)
        assert is_all_words_same_len

def test_has_correct_known_chars():
    res = model.predict(text)
    only_missing = res.get_only_predictions()
    expected_substrings=['ילדי?','???','י??ל']
    for i in range(len(only_missing)):
        txt_preds = [pred.value for pred in only_missing[i].predictions]
        for txt_pred in txt_preds:
            is_correct_known_chars(expected_substrings[i], txt_pred)

