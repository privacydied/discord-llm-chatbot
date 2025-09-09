# -*- coding: utf-8 -*-
import os
import platform
import sys
import importlib
import subprocess

import nltk
from nltk.corpus import cmudict

try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')

# These imports will fail until we copy the other files, that's expected for now.
# from g2pkk.special import jyeo, ye, consonant_ui, josa_ui, vowel_ui, jamo, rieulgiyeok, rieulbieub, verb_nieun, balb, palatalize, modifying_rieul
# from g2pkk.regular import link1, link2, link3, link4
# from g2pkk.utils import annotate, compose, group, gloss, parse_table, get_rule_id2text
# from g2pkk.english import convert_eng
# from g2pkk.numerals import convert_num


class G2p(object):
    def __init__(self):
        # self.check_mecab() # PATCHED: This is the root cause of the crash.
        self.mecab = self.get_mecab()
        self.table = None # parse_table()

        self.cmu = cmudict.dict() # for English

        self.rule2text = {} # get_rule_id2text() # for comments of main rules
        self.idioms_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idioms.txt")

    def load_module_func(self, module_name):
        tmp = __import__(module_name, fromlist=[module_name])
        return tmp

    def check_mecab(self):
        if platform.system()=='Windows':
            spam_spec = importlib.util.find_spec("eunjeon")
            non_found = spam_spec is None
            if non_found:
                print('you have to install eunjeon. install it...')
                p = subprocess.Popen('pip install eunjeon')
                p.wait()
        else:
            spam_spec = importlib.util.find_spec("mecab")
            non_found = spam_spec is None
            if non_found:
                print('you have to install python-mecab-ko. install it...')
                p = subprocess.Popen([sys.executable, "-m", "pip", "install", 'python-mecab-ko'])
                p.wait()


    def get_mecab(self):
        if platform.system() == 'Windows':
            try:
                m = self.load_module_func('eunjeon')
                return m.Mecab()
            except Exception:
                raise print('you have to install eunjeon. "pip install eunjeon"')
        else:
            try:
                m = self.load_module_func('mecab')
                return m.MeCab()
            except Exception:
                print('you have to install python-mecab-ko. "pip install python-mecab-ko"')


    def idioms(self, string, descriptive=False, verbose=False):
        return string # Patched to not require idioms.txt for now

    def __call__(self, string, descriptive=False, verbose=False, group_vowels=False, to_syl=True):
        # Dummy implementation, as the original requires all the other files.
        # The goal is just to make the library importable and not crash.
        print("Patched g2pkk called!")
        return string
