from src.classes.model_result import ModelResult
from src.classes.prediction import Prediction
from src.classes.text_part import TextPart
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.model.stub_model import StubModel
from src.evaluation.Classes.MRR_Chars import MRR_CHARS


def test_predict1():
    chak = MRR_CHARS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data) == 1

def test_predict2():
    chak = MRR_CHARS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 0.9), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data) == 0.9

def test_predict3():
    chak = MRR_CHARS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 0.8), Prediction('עוהב', 1)]),
        TextPart('?', [Prediction('ועוגות', 0.9), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data) == 0.8


def test_predict4():
    chak = MRR_CHARS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('שונא', 1), Prediction('יודע', 0.3)]),
        TextPart('?', [ Prediction('ומאכלי', 0.9)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data) == 0.2


def test_predict5():
    chak = MRR_CHARS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('אוהב', 1), Prediction('עוגב', 1)]),
        TextPart('?', [Prediction('ועוגות', 1), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע???ת גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו", "19": "ג", "20": "ו"}}]
    assert chak.calculate(model, data) == 1

def test_predict6():
    chak = MRR_CHARS()
    model = StubModel(ModelResult([
        TextPart('?', [Prediction('בגדה', 1), Prediction('עבגד', 1)]),
        TextPart('?', [Prediction('ועמגות', 1), Prediction('ומאפים', 1)]),
    ]))
    data = [{"text": "אני ???? שוקולד וע?גות גבינה", "missing": {"4": "א", "5": "ו", "6": "ה", "7": "ב", "18": "ו"}}]
    assert chak.calculate(model, data) == 0