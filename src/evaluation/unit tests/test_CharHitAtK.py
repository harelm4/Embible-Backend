from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.model.stub_model import StubModel


def test_predict1():
    chak = CharHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data, 5) == 1

def test_predict2():
    chak = CharHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 0.9), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data, 1) == 0.8



def test_predict3():
    chak = CharHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפים', 0.9)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data, 1) == 1


def test_predict4():
    chak = CharHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('שונא', 1), Prediction('יודע', 0.3)]),
        TextPart('?', [ Prediction('ומאכלי', 0.9)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data, 1) == 0.2

def test_model_result_to_list_of_preds():
    chak = CharHitAtK()
    modelRes = ModelResult([
        TextPart('?', [Prediction('מיל1', 1), Prediction('מיל2', 0.3)]),
        TextPart('?', [ Prediction('מילה33', 0.9)]),
    ])
    res = chak._model_result_to_list_of_preds(modelRes)
    assert res ==[['מיל1','מיל2'],['מילה33']]

def test_get_missing_idxs():
    chak = CharHitAtK()
    text='אני ה?לך לנ?ן'
    assert chak._get_missing_idxs(0,text)==[1]
    assert chak._get_missing_idxs(1, text) == [2]



test_predict1()
test_predict2()
test_predict3()
test_model_result_to_list_of_preds()
test_get_missing_idxs()