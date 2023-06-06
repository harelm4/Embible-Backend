class TextPart:
    def __init__(self,text,predictions):
        self.text=text
        self.predictions=predictions

    def __str__(self):
        return f'{self.text} : {self.predictions}'

    def is_only_text(self)->bool:
        return (self.predictions==None)

    def sorted(self,order='desc'):
        if order=='desc':
            self.predictions=list(sorted(self.predictions, key=lambda x: x.score, reverse=True))
        if order=='asc':
            self.predictions = list(sorted(self.predictions, key=lambda x: x.score))
        return self