from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.model.stub_model import StubModel


def test_predict1():
    whak = WordHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data, 5) == 1

def test_predict2():
    whak = WordHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגיות', 1), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data, 5) == 0.5



def test_predict3():
    whak = WordHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 0,9), Prediction('ומאפי', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data, 1) == 0.5


def test_predict4():
    whak = WordHitAtK()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('שנא', 1), Prediction('ידע', 0.3)]),
        TextPart('?', [ Prediction('ומאכלי', 0.9)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert whak.calculate(model, data, 5) == 0
