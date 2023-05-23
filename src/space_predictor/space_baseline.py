from src.space_predictor.space_predictor import space_predictor


class space_baseline(space_predictor):
    def __init__(self):
        # average word len in the bible
        self.avg_len=4

    def genText(self,text:str,threshold:int=0.5)->str:
        res_lst=[]
        splited = text.split()
        for part_idx,part in enumerate(splited):
            part_lst = list(part)
            if len(part)>self.avg_len:
                for i in range(self.avg_len,len(part_lst),self.avg_len+1):
                    if part_lst[i]=='?':
                        part_lst[i]=' '
            res_lst.extend(part_lst)
            if part_idx+1!=len(splited):
                res_lst.append(' ')
        return ''.join(res_lst)

