import json
from collections import defaultdict
from os import walk

import pandas as pd

df = pd.read_excel('../../data/train_df_no_niqqud.xlsx')
lens=[]
for verse in df['verse'].tolist():
    lens+=[len(x) for x in verse.split()]

print(f'avg:{round(sum(lens)/len(lens))}')
