import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.model.ensemble_v2 import EnsembleV2

res=[]
ensemble = EnsembleV2()
mix_file = f'../../data/Hit@K/masked MIX char tokens/mix_15.json'
char_hit_at_k=CharHitAtK()
data=char_hit_at_k.get_data_at_hit_at_k_test_format(mix_file)
rng=np.arange(0, 1, 0.1)
for w in rng:
    score = char_hit_at_k.calculate(ensemble, data, 5,char_weight=w)
    print(score)
    res.append(score)

plt.bar(rng,res)
plt.show()