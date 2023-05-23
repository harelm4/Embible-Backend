from typing import List

from data.test import space_predictor
from src.evaluation.Classes.Metric import Metric


class SpacePrecision(Metric):
    def calculate(self,predictor:space_predictor,data: str or List[dict],k:int)->float:
        pass