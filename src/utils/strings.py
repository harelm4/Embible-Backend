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
        words = s['text'].split()
        indeces=[]
        char_inex=0
        for i, word in enumerate(words):
            if all(c == '?' for c in word):
                indeces.append((char_inex,len(word)))
            char_inex+=len(word)+1
        return indeces

    @staticmethod
    def index_of_words_contains_question_mark(s:str):
        """

        :param s: string-text
        :return: indeces of all the words in the sentence that contains a ?
        """
        words = s.split()
        indeces=[]
        for i, word in enumerate(words):
            if any(c == '?' for c in word):
                indeces.append((i,len(word)))
        return indeces

    @staticmethod
    def get_known_chars_and_their_indeces(s:str):
        '''
        :param s: input text
        :return: dict: {index_of_masked_word:[(index of char in word,char)]}
        '''
        words = s.split()
        indeces = {}
        for i, word in enumerate(words):
            if any(c == '?' for c in word):
                for j,c in enumerate(word):
                    if(i in indeces.keys()):
                        indeces[i]=[(j,c)]#j- index in word, c-the char itself
                    else:
                        indeces[i].append((j,c))
        return indeces