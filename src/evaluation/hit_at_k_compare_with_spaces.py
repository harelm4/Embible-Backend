from typing import List

import pandas as pd

from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.evaluation.Classes.HitAtK import HitAtK
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.evaluation.hit_at_k_compare import  data_masked1, data_masked2, hit_at_k_eval
from src.model.ensemble_v2 import EnsembleV2
from src.space_predictor.Iterative_space_predictor import Iterative_space_predictor
from src.space_predictor.Recursive_space_predictor import Recursive_space_predictor
# TODO
word_hit_at_k = WordHitAtK()
char_hit_at_k = CharHitAtK()

it_ens=EnsembleV2(Iterative_space_predictor())
it_ens.model_path='EnsembleV2_iterative_sp'
rec_ens=EnsembleV2(Recursive_space_predictor())
rec_ens.model_path='EnsembleV2_recurrent_sp'
ensembles =[it_ens,rec_ens]

def hit_at_k_sp( k: int, hit_at_k_strategy: HitAtK):
    results = []
    for i, mask_precent in enumerate(data_masked1):
        mix_file = f'../../data/Hit@K/masked chars and subwords no masked spaces char tokens/masked_test_df_chars_and_subwords_no_masked_spaces_char_tokens_{masking_lst[i]}_no_niqqud_new.json'
        for model in ensembles:
            hit_at_k_eval(model, mix_file, k, hit_at_k_strategy, results)

        res_df = pd.DataFrame(results)
        csv_location = '../../data/results/' + f'Space-Pred/{hit_at_k_strategy.__class__.__name__[:-1]}{k}.csv'
        print(f'writing to {csv_location}')
        res_df.to_csv(csv_location)



for k in [1,5]:
    for strategy in [char_hit_at_k,word_hit_at_k]:
        hit_at_k_sp( k, strategy)
