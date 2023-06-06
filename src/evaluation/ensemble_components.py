import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.model.Char_model import CharModel
from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel
from src.model.ensemble_union import EnsembleUnion
from src.model.ensemble_v2 import EnsembleV2

ensemble = EnsembleV2()
char_model = CharModel(config.configs['char_model_path'])
word_model = SameLengthAndCharsWordModel(config.configs['word_model_path'])
ensemble_union = EnsembleUnion()
char_hit_at_k = CharHitAtK()
res = []
for p in [5, 15, 25, 30]:
    for k in [1,5]:
        for metric in [CharHitAtK(),WordHitAtK()]:
            mix_file = f'../../data/Hit@K/no masked spaces char tokens/{p}.json'
            scores = {}
            data = char_hit_at_k.get_data_at_hit_at_k_test_format(mix_file)
            scores['metric'] = metric.__class__.__name__
            scores['hide_p'] = p
            scores['k'] = k
            scores['ensemble_union'] = metric.calculate(ensemble_union, data, k)
            scores['ensemble'] = metric.calculate(ensemble, data, k)
            scores['char_model'] = metric.calculate(char_model, data, k)
            scores['word_model'] = metric.calculate(word_model, data, k)
            res.append(scores)
            print(scores)
df = pd.DataFrame(res)
print(df)
csv_location = '../../data/results/ensemble-component-compare.csv'
df.to_csv(csv_location)
