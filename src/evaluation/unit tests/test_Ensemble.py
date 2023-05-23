import time
from typing import List

import pandas as pd
from huggingface_hub import login
from src.model.Char_model import CharModel
import config
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.evaluation.Classes.Metric import Metric
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.model.SameLengthWordModel import SameLengthWordModel
from src.model.ensemble_v2 import EnsembleV2
from src.model.model import Model
from src.model.word_model import WordModel
from src.space_predictor.Iterative_space_predictor import Iterative_space_predictor
word_hit_at_k = WordHitAtK()
char_hit_at_k = CharHitAtK()
login(config.configs['hf_token'])

ensemble = EnsembleV2()
char_model=model = CharModel(config.configs['char_model_path'])
def test_with_spaces():
    text="נח?ו נחמו???? יאמר אלהיכם"
    missings={"2":"מ","10":"ע","11":"מ","12":"י","9":" "}
    space_predictor=Iterative_space_predictor()
    x=ensemble.predict(text,space_predictor=space_predictor)
    print(x)

test_with_spaces()