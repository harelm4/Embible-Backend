import pandas as pd

from src.evaluation.Classes.space_precision import space_precision
from src.evaluation.Classes.space_recall import space_recall
from src.space_predictor.Iterative_space_predictor import Iterative_space_predictor
from src.space_predictor.Recursive_space_predictor import Recursive_space_predictor
from src.space_predictor.space_baseline import space_baseline

predictors=[space_baseline(),Iterative_space_predictor(),Recursive_space_predictor()]
metrics=[space_precision(), space_recall()]
mask_lst = [5, 10, 15]


def get_data_at_test_format(file_path: str) -> list:
    """
    A function that coverts data from data foldr format to the formerly used format
    :param file_path: test file path
    :return: list in form of [{"text": "...", "missing": {...}}]
    """
    df = pd.read_json(file_path, orient='records', lines=True)
    data = df.to_dict(orient='records')
    return [{'text': x['verse'], 'missing': x['missing_dictionary']} for x in data]

results = []
for mask in mask_lst:
    file = f'../../data/Hit@K/masked MIX char tokens/mix_{mask}.json'
    data = get_data_at_test_format(file)
    for predictor in predictors:
        for metric in metrics:
            metric_res = metric.calculate(predictor,data)
            toAdd={'predictor':predictor.__class__.__name__,'file':file,'metric':metric.__class__.__name__.split('_')[1],'score':metric_res}
            results.append(toAdd)
            print(toAdd)
        res_df = pd.DataFrame(results)

csv_location = '../../data/results/space-predictor-eval.csv'
print(f'writing to {csv_location}')
res_df.to_csv(csv_location)