import re
from typing import List


class StringUtils():

    @staticmethod
    def  insert_masks(text:str)->str:
        res = ''
        for letter in text:
            add = ''
            if letter == '?':
                add = '[MASK]'
            else:
                add = letter
            res += add
        return res

    @staticmethod
    def split_text(text:str)->List[str]:
        return list(filter(lambda x:x!='',re.split('(\?)',text)))

