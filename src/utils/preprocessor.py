from hebrew.chars import CHARS
import json
from os import walk

class Preprocessor():
    
    def text_to_json(self,file_name):
        raw_text=''
        with open('data/bible_books/{name}'.format(name=file_name),encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            raw_text+=line
        char_lst=list(CHARS.keys())
        char_lst.append(' ')
        tmp=''
        pre_res=[]
        for char in raw_text:
            if len(tmp) !=0 and tmp[-1]==' ' and char==' ':
                continue
            if char=='׃':
                pre_res.append(tmp)
                tmp=''
                continue
            if char=='־':
                tmp+=' '
                continue
            if char.encode("UTF-8").decode("UTF-8") in char_lst:
                tmp+=char
        res=[]
        for verse in pre_res:

            if verse not in [' ׃',' פ ',' ס ',' ',' ו ','ס','׀','פ',', ',' ,']:
                res.append(verse)

        with open('data/bible_books_jsons/{name}.json'.format(name=file_name), 'w',encoding="utf8") as f:
            json.dump(res, f,ensure_ascii=False)

    def parse_all_bible_books_to_json_lists(self):
        filenames = next(walk('data/bible_books'), (None, None, []))[2]  # [] if no file
        for file in filenames:
            self.text_to_json(file)
    def get_all_words_in_jsons(self):
        pre_res=[]
        res=[]
        filenames = next(walk('data/bible_books'), (None, None, []))[2]  # [] if no file
        for file_name in filenames:
            with open('data/bible_books_jsons/{name}.json'.format(name=file_name), encoding="utf8") as f:
                for p in f.readlines():
                    pre_res+=p.split(' ')
        for word in pre_res:
            to_add=True
            for char in word:
                if char not in CHARS:
                    to_add=False
            if to_add:
                res.append(word)
            to_add = True
        return res

    def resolveSOP(self):

        filenames = next(walk('data/bible_books'), (None, None, []))[2]  # [] if no file
        for file_name in filenames:
            with open('data/bible_books_jsons/{name}.json'.format(name=file_name), encoding="utf8") as f:
                new_verses = []
                for verse in json.load(f):
                    verse_without_unwanted=[]
                    for word in verse.split(' '):
                        if len(word)>1 and ',' not in word:
                            verse_without_unwanted.append(word)
                    new_verses.append(' '.join(verse_without_unwanted))

            with open('data/bible_books_jsons/{name}.json'.format(name=file_name), 'w', encoding="utf8") as f:
                json.dump(new_verses, f, ensure_ascii=False)



