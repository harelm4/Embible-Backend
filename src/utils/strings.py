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

    @staticmethod
    def change_question_mark_in_text(text:str)->List[str]:
        """
        the function changes each word that contains ? to ?
        :param text:
        :return: new string
        """
        words = text.split()
        output_words = []
        lens_of_masked_words=[]
        for word in words:
            if '?' in word:
                output_words.append('?')
                lens_of_masked_words.append(len(word))
            else:
                output_words.append(word)
        return ' '.join(output_words),lens_of_masked_words

    @staticmethod
    def find_question_word_index(s):
        words = s['text'].split()  # split the string into words
        indeces=[]
        char_inex=0
        for i, word in enumerate(words):
            if all(c == '?' for c in word):
                indeces.append((char_inex,len(word)))  # return the index of the first question word
            char_inex+=len(word)+1
        return indeces  # return -1 if no question word is found


