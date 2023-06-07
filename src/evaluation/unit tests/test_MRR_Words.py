from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.evaluation.Classes.MRR_Words import MRR_WORDS
from src.model.stub_model import StubModel


def test_predict1():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data) == 1

def test_predict2():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגיות', 1), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data) == 0.5

def test_predict3():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 0.75), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגיות', 1), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data) == 0.25

def test_predict4():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 0.25), Prediction('עוהב', 1), Prediction('שובר', 0.75)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפי', 0.5)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert round(whak.calculate(model, data),2) == 0.67
def test_predict5():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 0.9), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data) == 0.75


def test_predict7():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('שנא', 1), Prediction('ידע', 0.3)]),
        TextPart('?', [ Prediction('ומאכלי', 0.9)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data) == 0


def test_predict8():
    whak = MRR_WORDS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע???ת גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו", "19": "ג", "20": "ו"}}]
    assert whak.calculate(model, data) == 1


