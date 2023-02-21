from src.classes.text_part import TextPart


class EnsembleResult:
    def __init__(self,lst:list):
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
    def __str__(self):
        return str(self.get_ui_format())