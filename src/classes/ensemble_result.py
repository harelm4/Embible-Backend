from src.classes.text_part import TextPart
from typing import List

class EnsembleResult:
    def __init__(self,lst:List[TextPart]):
        self.lst=lst
    def get_ui_format(self):
        res=[]
        for tp in self.lst:
            if not tp.is_only_text():
                new_tp={'text':tp.text,'predictions':[]}
                for pred in tp.predictions:
                    new_tp['predictions'].append({'value':pred.value,'p':pred.score})
            else:
                new_tp={'text':tp.text,'predictions':None}
            res.append(new_tp)
        return res

    def get_only_predictions(self):
        return EnsembleResult([x for x in self.lst if x.text=='?'])
    def get_only_k_predictions(self,k):
        return EnsembleResult([TextPart(x.text,x.predictions[0:k]) for x in self.lst if x.text=='?'])
    def __str__(self):
        return str(self.get_ui_format())