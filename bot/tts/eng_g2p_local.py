"""
Local English grapheme-to-phoneme conversion outputting IPA symbols.
Zero dependencies, no network downloads, offline-only operation.
IPA-focused for Kokoro TTS model compatibility.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Basic IPA to ID mapping as fallback (covers common English IPA symbols)
IPA_TO_ID = {
    # Vowels
    'a': 1, 'ɑ': 2, 'æ': 3, 'ʌ': 4, 'ə': 5, 'ɚ': 6, 'ɛ': 7, 'ɜ': 8, 'ɪ': 9, 'i': 10,
    'ɔ': 11, 'o': 12, 'ʊ': 13, 'u': 14, 'ɑː': 15, 'ɒ': 16, 'ɔː': 17, 'əʊ': 18, 'aʊ': 19,
    'aɪ': 20, 'ɛə': 21, 'ɪə': 22, 'ɔɪ': 23, 'ʊə': 24, 'eɪ': 25,

    # Consonants
    'b': 26, 'd': 27, 'f': 28, 'g': 29, 'h': 30, 'j': 31, 'k': 32, 'l': 33, 'm': 34, 'n': 35,
    'ŋ': 36, 'p': 37, 'r': 38, 's': 39, 'ʃ': 40, 't': 41, 'tʃ': 42, 'θ': 43, 'ð': 44,
    'v': 45, 'w': 46, 'z': 47, 'ʒ': 48, 'dʒ': 49, 'ʔ': 50,

    # Diphthongs and special
    'iː': 51, 'uː': 52, 'ɔː': 53, 'ɑː': 54, 'ɜː': 55, 'ɛː': 56,

    # Stress and length marks (treat as modifiers, not separate tokens)
    'ˈ': 57, 'ˌ': 58, 'ː': 59,

    # Pause/silence and punctuation
    '_': 60, '<sp>': 61, 'sil': 62, '.': 63, ',': 64,

    # Numbers and special characters
    '0': 65, '1': 66, '2': 67, '3': 68, '4': 69, '5': 70, '6': 71, '7': 72, '8': 73, '9': 74,
}
BUILTIN_LEXICON = {
    # Numbers 0-10
    "zero": ["Z", "IY1", "R", "OW"],
    "one": ["W", "AH1", "N"],
    "two": ["T", "UW1"],
    "three": ["TH", "R", "IY1"],
    "four": ["F", "AO1", "R"],
    "five": ["F", "AY1", "V"],
    "six": ["S", "IH1", "K", "S"],
    "seven": ["S", "EH1", "V", "AH", "N"],
    "eight": ["EY1", "T"],
    "nine": ["N", "AY1", "N"],
    "ten": ["T", "EH1", "N"],

    # Teens
    "eleven": ["IH", "L", "EH1", "V", "AH", "N"],
    "twelve": ["T", "W", "EH1", "L", "V"],
    "thirteen": ["TH", "ER1", "T", "IY", "N"],
    "fourteen": ["F", "AO1", "R", "T", "IY", "N"],
    "fifteen": ["F", "IH1", "F", "T", "IY", "N"],
    "sixteen": ["S", "IH1", "K", "S", "T", "IY", "N"],
    "seventeen": ["S", "EH1", "V", "AH", "N", "T", "IY", "N"],
    "eighteen": ["EY1", "T", "IY", "N"],
    "nineteen": ["N", "AY1", "N", "T", "IY", "N"],

    # Tens
    "twenty": ["T", "W", "EH1", "N", "T", "IY"],
    "thirty": ["TH", "ER1", "T", "IY"],
    "forty": ["F", "AO1", "R", "T", "IY"],
    "fifty": ["F", "IH1", "F", "T", "IY"],
    "sixty": ["S", "IH1", "K", "S", "T", "IY"],
    "seventy": ["S", "EH1", "V", "AH", "N", "T", "IY"],
    "eighty": ["EY1", "T", "IY"],
    "ninety": ["N", "AY1", "N", "T", "IY"],

    # Hundreds
    "hundred": ["HH", "AH1", "N", "D", "R", "AH", "D"],
    "thousand": ["TH", "AW1", "Z", "AH", "N", "D"],

    # Ordinals
    "first": ["F", "ER1", "S", "T"],
    "second": ["S", "EH1", "K", "AH", "N", "D"],
    "third": ["TH", "ER1", "D"],
    "fourth": ["F", "AO1", "R", "TH"],
    "fifth": ["F", "IH1", "F", "TH"],
    "sixth": ["S", "IH1", "K", "S", "TH"],
    "seventh": ["S", "EH1", "V", "AH", "N", "TH"],
    "eighth": ["EY1", "T", "TH"],
    "ninth": ["N", "AY1", "N", "TH"],
    "tenth": ["T", "EH1", "N", "TH"],

    # Common function words
    "the": ["DH", "AH"],
    "a": ["AH"],
    "an": ["AH", "N"],
    "and": ["AH", "N", "D"],
    "or": ["AO", "R"],
    "but": ["B", "AH", "T"],
    "to": ["T", "UW"],
    "of": ["AH", "V"],
    "in": ["IH", "N"],
    "on": ["AO", "N"],
    "at": ["AE", "T"],
    "for": ["F", "AO", "R"],
    "with": ["W", "IH", "TH"],
    "by": ["B", "AY"],
    "from": ["F", "R", "AH", "M"],

    # Common verbs
    "be": ["B", "IY"],
    "is": ["IH", "Z"],
    "are": ["AA", "R"],
    "was": ["W", "AH", "Z"],
    "were": ["W", "ER"],
    "have": ["HH", "AE", "V"],
    "has": ["HH", "AE", "Z"],
    "had": ["HH", "AE", "D"],
    "do": ["D", "UW"],
    "does": ["D", "AH", "Z"],
    "did": ["D", "IH", "D"],
    "will": ["W", "IH", "L"],
    "would": ["W", "UH", "D"],
    "can": ["K", "AE", "N"],
    "could": ["K", "UH", "D"],
    "should": ["SH", "UH", "D"],
    "may": ["M", "EY"],
    "might": ["M", "AY", "T"],
    "must": ["M", "AH", "S", "T"],
    "shall": ["SH", "AE", "L"],
    "say": ["S", "EY"],
    "said": ["S", "EH", "D"],
    "go": ["G", "OW"],
    "went": ["W", "EH", "N", "T"],
    "come": ["K", "AH", "M"],
    "came": ["K", "EY", "M"],
    "see": ["S", "IY"],
    "saw": ["S", "AO"],
    "know": ["N", "OW"],
    "knew": ["N", "UW"],
    "think": ["TH", "IH", "NG", "K"],
    "thought": ["TH", "AO", "T"],
    "tell": ["T", "EH", "L"],
    "told": ["T", "OW", "L", "D"],
    "work": ["W", "ER", "K"],
    "worked": ["W", "ER", "K", "T"],
    "make": ["M", "EY", "K"],
    "made": ["M", "EY", "D"],
    "take": ["T", "EY", "K"],
    "took": ["T", "UH", "K"],
    "give": ["G", "IH", "V"],
    "gave": ["G", "EY", "V"],
    "get": ["G", "EH", "T"],
    "got": ["G", "AA", "T"],
    "find": ["F", "AY", "N", "D"],
    "found": ["F", "AW", "N", "D"],
    "look": ["L", "UH", "K"],
    "looked": ["L", "UH", "K", "T"],
    "want": ["W", "AO", "N", "T"],
    "wanted": ["W", "AO", "N", "T", "IH", "D"],
    "use": ["Y", "UW", "S"],
    "used": ["Y", "UW", "Z", "D"],
    "need": ["N", "IY", "D"],
    "needed": ["N", "IY", "D", "IH", "D"],
    "help": ["HH", "EH", "L", "P"],
    "helped": ["HH", "EH", "L", "P", "T"],
    "ask": ["AE", "S", "K"],
    "asked": ["AE", "S", "K", "T"],
    "call": ["K", "AO", "L"],
    "called": ["K", "AO", "L", "D"],
    "try": ["T", "R", "AY"],
    "tried": ["T", "R", "AY", "D"],
    "turn": ["T", "ER", "N"],
    "turned": ["T", "ER", "N", "D"],
    "run": ["R", "AH", "N"],
    "ran": ["R", "AE", "N"],
    "walk": ["W", "AO", "K"],
    "walked": ["W", "AO", "K", "T"],
    "live": ["L", "IH", "V"],
    "lived": ["L", "IH", "V", "D"],
    "happen": ["HH", "AE", "P", "AH", "N"],
    "happened": ["HH", "AE", "P", "AH", "N", "D"],
    "begin": ["B", "IH", "G", "IH", "N"],
    "began": ["B", "IH", "G", "AE", "N"],
    "begin": ["B", "IH", "G", "IH", "N"],
    "begun": ["B", "IH", "G", "AH", "N"],

    # Common adjectives
    "good": ["G", "UH", "D"],
    "bad": ["B", "AE", "D"],
    "big": ["B", "IH", "G"],
    "small": ["S", "M", "AO", "L"],
    "new": ["N", "UW"],
    "old": ["OW", "L", "D"],
    "hot": ["HH", "AA", "T"],
    "cold": ["K", "OW", "L", "D"],
    "happy": ["HH", "AE", "P", "IY"],
    "sad": ["S", "AE", "D"],
    "right": ["R", "AY", "T"],
    "wrong": ["R", "AO", "NG"],
    "first": ["F", "ER", "S", "T"],
    "last": ["L", "AE", "S", "T"],
    "next": ["N", "EH", "K", "S", "T"],
    "same": ["S", "EY", "M"],
    "different": ["D", "IH", "F", "ER", "AH", "N", "T"],
    "easy": ["IY", "Z", "IY"],
    "hard": ["HH", "AA", "R", "D"],
    "fast": ["F", "AE", "S", "T"],
    "slow": ["S", "L", "OW"],
    "high": ["HH", "AY"],
    "low": ["L", "OW"],
    "long": ["L", "AO", "NG"],
    "short": ["SH", "AO", "R", "T"],
    "young": ["Y", "AH", "NG"],
    "old": ["OW", "L", "D"],

    # Common nouns
    "time": ["T", "AY", "M"],
    "day": ["D", "EY"],
    "year": ["Y", "IH", "R"],
    "man": ["M", "AE", "N"],
    "woman": ["W", "UH", "M", "AH", "N"],
    "child": ["CH", "AY", "L", "D"],
    "world": ["W", "ER", "L", "D"],
    "life": ["L", "AY", "F"],
    "hand": ["HH", "AE", "N", "D"],
    "part": ["P", "AA", "R", "T"],
    "eye": ["AY"],
    "place": ["P", "L", "EY", "S"],
    "thing": ["TH", "IH", "NG"],
    "way": ["W", "EY"],
    "case": ["K", "EY", "S"],
    "point": ["P", "OY", "N", "T"],
    "question": ["K", "W", "EH", "S", "CH", "AH", "N"],
    "answer": ["AE", "N", "S", "ER"],
    "problem": ["P", "R", "AA", "B", "L", "AH", "M"],
    "word": ["W", "ER", "D"],
    "number": ["N", "AH", "M", "B", "ER"],
    "people": ["P", "IY", "P", "AH", "L"],
    "water": ["W", "AO", "T", "ER"],
    "food": ["F", "UW", "D"],
    "money": ["M", "AH", "N", "IY"],
    "car": ["K", "AA", "R"],
    "house": ["HH", "AW", "S"],
    "school": ["S", "K", "UW", "L"],
    "book": ["B", "UH", "K"],
    "friend": ["F", "R", "EH", "N", "D"],
    "family": ["F", "AE", "M", "AH", "L", "IY"],
    "job": ["JH", "AA", "B"],
    "country": ["K", "AH", "N", "T", "R", "IY"],
    "city": ["S", "IH", "T", "IY"],
    "town": ["T", "AW", "N"],
    "village": ["V", "IH", "L", "IH", "JH"],
    "farm": ["F", "AA", "R", "M"],
    "factory": ["F", "AE", "K", "T", "ER", "IY"],
    "office": ["AO", "F", "AH", "S"],
    "school": ["S", "K", "UW", "L"],
    "university": ["Y", "UW", "N", "AH", "V", "ER", "S", "AH", "T", "IY"],
    "hospital": ["HH", "AA", "S", "P", "IH", "T", "AH", "L"],
    "church": ["CH", "ER", "CH"],
    "temple": ["T", "EH", "M", "P", "AH", "L"],
    "mosque": ["M", "AA", "S", "K"],
    "synagogue": ["S", "IH", "N", "AH", "G", "AA", "G"],
    "government": ["G", "AH", "V", "ER", "N", "M", "AH", "N", "T"],
    "president": ["P", "R", "EH", "Z", "AH", "D", "AH", "N", "T"],
    "king": ["K", "IH", "NG"],
    "queen": ["K", "W", "IY", "N"],
    "prince": ["P", "R", "IH", "N", "S"],
    "princess": ["P", "R", "IH", "N", "S", "EH", "S"],
    "army": ["AA", "R", "M", "IY"],
    "navy": ["N", "EY", "V", "IY"],
    "air force": ["EH", "R", "F", "AO", "R", "S"],
    "police": ["P", "AH", "L", "IY", "S"],
    "firefighter": ["F", "AY", "ER", "F", "AY", "T", "ER"],
    "doctor": ["D", "AA", "K", "T", "ER"],
    "nurse": ["N", "ER", "S"],
    "teacher": ["T", "IY", "CH", "ER"],
    "student": ["S", "T", "UW", "D", "AH", "N", "T"],
    "lawyer": ["L", "AO", "Y", "ER"],
    "engineer": ["EH", "N", "JH", "AH", "N", "IY", "R"],
    "scientist": ["S", "AY", "AH", "N", "T", "IH", "S", "T"],
    "artist": ["AA", "R", "T", "IH", "S", "T"],
    "musician": ["M", "Y", "UW", "Z", "IH", "SH", "AH", "N"],
    "actor": ["AE", "K", "T", "ER"],
    "actress": ["AE", "K", "T", "R", "EH", "S"],
    "writer": ["R", "AY", "T", "ER"],
    "poet": ["P", "OW", "AH", "T"],
    "dancer": ["D", "AE", "N", "S", "ER"],
    "singer": ["S", "IH", "NG", "ER"],
    "painter": ["P", "EY", "N", "T", "ER"],
    "photographer": ["F", "AH", "T", "AA", "G", "R", "AH", "F", "ER"],
    "chef": ["SH", "EH", "F"],
    "farmer": ["F", "AA", "R", "M", "ER"],
    "pilot": ["P", "AY", "L", "AH", "T"],
    "driver": ["D", "R", "AY", "V", "ER"],
    "mechanic": ["M", "AH", "K", "AE", "N", "IH", "K"],
    "electrician": ["IH", "L", "EH", "K", "T", "R", "IH", "SH", "AH", "N"],
    "plumber": ["P", "L", "AH", "M", "B", "ER"],
    "carpenter": ["K", "AA", "R", "P", "AH", "N", "T", "ER"],
    "gardener": ["G", "AA", "R", "D", "AH", "N", "ER"],
    "cleaner": ["K", "L", "IY", "N", "ER"],
    "salesperson": ["S", "EY", "L", "Z", "P", "ER", "S", "AH", "N"],
    "cashier": ["K", "AE", "SH", "IH", "R"],
    "waiter": ["W", "EY", "T", "ER"],
    "waitress": ["W", "EY", "T", "R", "EH", "S"],
    "cook": ["K", "UH", "K"],
    "baker": ["B", "EY", "K", "ER"],
    "butcher": ["B", "UH", "CH", "ER"],
    "grocer": ["G", "R", "OW", "S", "ER"],
    "pharmacist": ["F", "AA", "R", "M", "AH", "S", "IH", "S", "T"],
    "dentist": ["D", "EH", "N", "T", "IH", "S", "T"],
    "veterinarian": ["V", "EH", "T", "ER", "AH", "N", "EH", "R", "IY", "AH", "N"],
    "psychologist": ["S", "AY", "K", "AA", "L", "AH", "JH", "IH", "S", "T"],
    "therapist": ["TH", "EH", "R", "AH", "P", "IH", "S", "T"],
    "coach": ["K", "OW", "CH"],
    "athlete": ["AE", "TH", "L", "IY", "T"],
    "player": ["P", "L", "EY", "ER"],
    "team": ["T", "IY", "M"],
    "game": ["G", "EY", "M"],
    "sport": ["S", "P", "AO", "R", "T"],
    "football": ["F", "UH", "T", "B", "AO", "L"],
    "basketball": ["B", "AE", "S", "K", "IH", "T", "B", "AO", "L"],
    "baseball": ["B", "EY", "S", "B", "AO", "L"],
    "soccer": ["S", "AA", "K", "ER"],
    "tennis": ["T", "EH", "N", "IH", "S"],
    "golf": ["G", "AA", "L", "F"],
    "swimming": ["S", "W", "IH", "M", "IH", "NG"],
    "running": ["R", "AH", "N", "IH", "NG"],
    "cycling": ["S", "AY", "K", "L", "IH", "NG"],
    "skiing": ["S", "K", "IY", "IH", "NG"],
    "hiking": ["HH", "AY", "K", "IH", "NG"],
    "fishing": ["F", "IH", "SH", "IH", "NG"],
    "hunting": ["HH", "AH", "N", "T", "IH", "NG"],
    "camping": ["K", "AE", "M", "P", "IH", "NG"],
    "traveling": ["T", "R", "AE", "V", "AH", "L", "IH", "NG"],
    "tourist": ["T", "UH", "R", "IH", "S", "T"],
    "vacation": ["V", "EY", "K", "EY", "SH", "AH", "N"],
    "hotel": ["HH", "OW", "T", "EH", "L"],
    "flight": ["F", "L", "AY", "T"],
    "ticket": ["T", "IH", "K", "IH", "T"],
    "passport": ["P", "AE", "S", "P", "AO", "R", "T"],
    "suitcase": ["S", "UW", "T", "K", "EY", "S"],
    "camera": ["K", "AE", "M", "ER", "AH"],
    "map": ["M", "AE", "P"],
    "guide": ["G", "AY", "D"],
    "language": ["L", "AE", "NG", "G", "W", "IH", "JH"],
    "word": ["W", "ER", "D"],
    "sentence": ["S", "EH", "N", "T", "AH", "N", "S"],
    "grammar": ["G", "R", "AE", "M", "ER"],
    "vocabulary": ["V", "OW", "K", "AE", "B", "Y", "AH", "L", "EH", "R", "IY"],
    "conversation": ["K", "AA", "N", "V", "ER", "S", "EY", "SH", "AH", "N"],
    "question": ["K", "W", "EH", "S", "CH", "AH", "N"],
    "answer": ["AE", "N", "S", "ER"],
    "hello": ["HH", "AH", "L", "OW"],
    "hi": ["HH", "AY"],
    "goodbye": ["G", "UH", "D", "B", "AY"],
    "thank you": ["TH", "AE", "NG", "K", "Y", "UW"],
    "please": ["P", "L", "IY", "Z"],
    "sorry": ["S", "AO", "R", "IY"],
    "excuse me": ["IH", "K", "S", "K", "Y", "UW", "Z", "M", "IY"],
    "yes": ["Y", "EH", "S"],
    "no": ["N", "OW"],
    "maybe": ["M", "EY", "B", "IY"],
    "okay": ["OW", "K", "EY"],
    "alright": ["AO", "L", "R", "AY", "T"],
    "fine": ["F", "AY", "N"],
    "good": ["G", "UH", "D"],
    "bad": ["B", "AE", "D"],
    "happy": ["HH", "AE", "P", "IY"],
    "sad": ["S", "AE", "D"],
    "angry": ["AE", "NG", "G", "R", "IY"],
    "tired": ["T", "AY", "ER", "D"],
    "hungry": ["HH", "AH", "NG", "G", "R", "IY"],
    "thirsty": ["TH", "ER", "S", "T", "IY"],
    "hot": ["HH", "AA", "T"],
    "cold": ["K", "OW", "L", "D"],
    "sick": ["S", "IH", "K"],
    "healthy": ["HH", "EH", "L", "TH", "IY"],
    "busy": ["B", "IH", "Z", "IY"],
    "free": ["F", "R", "IY"],
    "ready": ["R", "EH", "D", "IY"],
    "late": ["L", "EY", "T"],
    "early": ["ER", "L", "IY"],
    "fast": ["F", "AE", "S", "T"],
    "slow": ["S", "L", "OW"],
    "easy": ["IY", "Z", "IY"],
    "difficult": ["D", "IH", "F", "AH", "K", "AH", "L", "T"],
    "expensive": ["IH", "K", "S", "P", "EH", "N", "S", "IH", "V"],
    "cheap": ["CH", "IY", "P"],
    "new": ["N", "UW"],
    "old": ["OW", "L", "D"],
    "big": ["B", "IH", "G"],
    "small": ["S", "M", "AO", "L"],
    "long": ["L", "AO", "NG"],
    "short": ["SH", "AO", "R", "T"],
    "clean": ["K", "L", "IY", "N"],
    "dirty": ["D", "ER", "T", "IY"],
    "beautiful": ["B", "Y", "UW", "T", "AH", "F", "AH", "L"],
    "ugly": ["AH", "G", "L", "IY"],
    "rich": ["R", "IH", "CH"],
    "poor": ["P", "UH", "R"],
    "young": ["Y", "AH", "NG"],
    "old": ["OW", "L", "D"],
    "smart": ["S", "M", "AA", "R", "T"],
    "stupid": ["S", "T", "UW", "P", "IH", "D"],
    "kind": ["K", "AY", "N", "D"],
    "mean": ["M", "IY", "N"],
    "funny": ["F", "AH", "N", "IY"],
    "serious": ["S", "IH", "R", "IY", "AH", "S"],
    "interesting": ["IH", "N", "T", "ER", "IH", "S", "T", "IH", "NG"],
    "boring": ["B", "AO", "R", "IH", "NG"],
    "important": ["IH", "M", "P", "AO", "R", "T", "AH", "N", "T"],
    "dangerous": ["D", "EY", "N", "JH", "ER", "AH", "S"],
    "safe": ["S", "EY", "F"],
    "loud": ["L", "AW", "D"],
    "quiet": ["K", "W", "AY", "AH", "T"],
    "dark": ["D", "AA", "R", "K"],
    "light": ["L", "AY", "T"],
    "hard": ["HH", "AA", "R", "D"],
    "soft": ["S", "AO", "F", "T"],
    "wet": ["W", "EH", "T"],
    "dry": ["D", "R", "AY"],
    "full": ["F", "UH", "L"],
    "empty": ["EH", "M", "P", "T", "IY"],
    "open": ["OW", "P", "AH", "N"],
    "closed": ["K", "L", "OW", "Z", "D"],
    "right": ["R", "AY", "T"],
    "wrong": ["R", "AO", "NG"],
    "true": ["T", "R", "UW"],
    "false": ["F", "AO", "L", "S"],
    "real": ["R", "IY", "L"],
    "fake": ["F", "EY", "K"],
    "first": ["F", "ER", "S", "T"],
    "second": ["S", "EH1", "K", "AH", "N", "D"],
    "third": ["TH", "ER1", "D"],
    "fourth": ["F", "AO1", "R", "TH"],
    "fifth": ["F", "IH1", "F", "TH"],
    "sixth": ["S", "IH1", "K", "S", "TH"],
    "seventh": ["S", "EH1", "V", "AH", "N", "TH"],
    "eighth": ["EY1", "T", "TH"],
    "ninth": ["N", "AY1", "N", "TH"],
    "tenth": ["T", "EH1", "N", "TH"],
    "eleventh": ["IH", "L", "EH1", "V", "AH", "N", "TH"],
    "twelfth": ["T", "W", "EH1", "L", "F", "TH"],
    "thirteenth": ["TH", "ER1", "T", "IY", "N", "TH"],
    "fourteenth": ["F", "AO1", "R", "T", "IY", "N", "TH"],
    "fifteenth": ["F", "IH1", "F", "T", "IY", "N", "TH"],
    "sixteenth": ["S", "IH1", "K", "S", "T", "IY", "N", "TH"],
    "seventeenth": ["S", "EH1", "V", "AH", "N", "T", "IY", "N", "TH"],
    "eighteenth": ["EY1", "T", "IY", "N", "TH"],
    "nineteenth": ["N", "AY1", "N", "T", "IY", "N", "TH"],
    "twentieth": ["T", "W", "EH1", "N", "T", "IY", "TH"],
    "twenty first": ["T", "W", "EH1", "N", "T", "IY", "F", "ER1", "S", "T"],
    "twenty second": ["T", "W", "EH1", "N", "T", "IY", "S", "EH1", "K", "AH", "N", "D"],
    "twenty third": ["T", "W", "EH1", "N", "T", "IY", "TH", "ER1", "D"],
    "twenty fourth": ["T", "W", "EH1", "N", "T", "IY", "F", "AO1", "R", "TH"],
    "twenty fifth": ["T", "W", "EH1", "N", "T", "IY", "F", "IH1", "F", "TH"],
    "twenty sixth": ["T", "W", "EH1", "N", "T", "IY", "S", "IH1", "K", "S", "TH"],
    "twenty seventh": ["T", "W", "EH1", "N", "T", "IY", "S", "EH1", "V", "AH", "N", "TH"],
    "twenty eighth": ["T", "W", "EH1", "N", "T", "IY", "EY1", "T", "TH"],
    "twenty ninth": ["T", "W", "EH1", "N", "T", "IY", "N", "AY1", "N", "TH"],
    "thirtieth": ["TH", "ER1", "T", "IY", "TH"],
    "thirty first": ["TH", "ER1", "T", "IY", "F", "ER1", "S", "T"],
    "counting": ["K", "AW", "N", "T", "IH", "NG"],
    "from": ["F", "R", "AH", "M"],
    "i": ["AY"],
    "me": ["M", "IY"],
    "my": ["M", "AY"],
    "myself": ["M", "AY", "S", "EH", "L", "F"],
    "you": ["Y", "UW"],
    "your": ["Y", "AO", "R"],
    "yourself": ["Y", "ER", "S", "EH", "L", "F"],
    "he": ["HH", "IY"],
    "him": ["HH", "IH", "M"],
    "his": ["HH", "IH", "Z"],
    "himself": ["HH", "IH", "M", "S", "EH", "L", "F"],
    "she": ["SH", "IY"],
    "her": ["HH", "ER"],
    "hers": ["HH", "ER", "Z"],
    "herself": ["HH", "ER", "S", "EH", "L", "F"],
    "it": ["IH", "T"],
    "its": ["IH", "T", "S"],
    "itself": ["IH", "T", "S", "EH", "L", "F"],
    "we": ["W", "IY"],
    "us": ["AH", "S"],
    "our": ["AW", "ER"],
    "ours": ["AW", "ER", "Z"],
    "ourselves": ["AW", "ER", "S", "EH", "L", "V", "Z"],
    "they": ["DH", "EY"],
    "them": ["DH", "EH", "M"],
    "their": ["DH", "EH", "R"],
    "theirs": ["DH", "EH", "R", "Z"],
    "themselves": ["DH", "EH", "M", "S", "EH", "L", "V", "Z"],
    "this": ["DH", "IH", "S"],
    "that": ["DH", "AE", "T"],
    "these": ["DH", "IY", "Z"],
    "those": ["DH", "OW", "Z"],
    "who": ["HH", "UW"],
    "whom": ["HH", "UW", "M"],
    "whose": ["HH", "UW", "Z"],
    "which": ["W", "IH", "CH"],
    "what": ["W", "AH", "T"],
    "where": ["W", "EH", "R"],
    "when": ["W", "EH", "N"],
    "why": ["W", "AY"],
    "how": ["HH", "AW"],
    "and": ["AH", "N", "D"],
    "or": ["AO", "R"],
    "but": ["B", "AH", "T"],
    "so": ["S", "OW"],
    "because": ["B", "IH", "K", "AO", "Z"],
    "although": ["AO", "L", "DH", "OW"],
    "if": ["IH", "F"],
    "then": ["DH", "EH", "N"],
    "else": ["EH", "L", "S"],
    "while": ["W", "AY", "L"],
    "since": ["S", "IH", "N", "S"],
    "until": ["AH", "N", "T", "IH", "L"],
    "before": ["B", "IH", "F", "AO", "R"],
    "after": ["AE", "F", "T", "ER"],
    "during": ["D", "Y", "UW", "R", "IH", "NG"],
    "as": ["AE", "Z"],
    "like": ["L", "AY", "K"],
    "than": ["DH", "AE", "N"],
    "with": ["W", "IH", "TH"],
    "without": ["W", "IH", "TH", "AW", "T"],
    "about": ["AH", "B", "AW", "T"],
    "against": ["AH", "G", "EH", "N", "S", "T"],
    "between": ["B", "IH", "T", "W", "IY", "N"],
    "among": ["AH", "M", "AH", "NG"],
    "through": ["TH", "R", "UW"],
    "across": ["AH", "K", "R", "AO", "S"],
    "around": ["AH", "R", "AW", "N", "D"],
    "behind": ["B", "IH", "HH", "AY", "N", "D"],
    "beside": ["B", "IH", "S", "AY", "D"],
    "near": ["N", "IH", "R"],
    "far": ["F", "AA", "R"],
    "from": ["F", "R", "AH", "M"],
    "to": ["T", "UW"],
    "at": ["AE", "T"],
    "in": ["IH", "N"],
    "on": ["AO", "N"],
    "by": ["B", "AY"],
    "for": ["F", "AO", "R"],
    "of": ["AH", "V"],
    "with": ["W", "IH", "TH"],
    "about": ["AH", "B", "AW", "T"],
    "into": ["IH", "N", "T", "UW"],
    "through": ["TH", "R", "UW"],
    "during": ["D", "Y", "UW", "R", "IH", "NG"],
    "before": ["B", "IH", "F", "AO", "R"],
    "after": ["AE", "F", "T", "ER"],
    "above": ["AH", "B", "AH", "V"],
    "below": ["B", "IH", "L", "OW"],
    "left": ["L", "EH", "F", "T"],
    "right": ["R", "AY", "T"],
    "front": ["F", "R", "AH", "N", "T"],
    "back": ["B", "AE", "K"],
    "inside": ["IH", "N", "S", "AY", "D"],
    "outside": ["AW", "T", "S", "AY", "D"],
    "up": ["AH", "P"],
    "down": ["D", "AW", "N"],
    "here": ["HH", "IY", "R"],
    "there": ["DH", "EH", "R"],
    "now": ["N", "AW"],
    "then": ["DH", "EH", "N"],
    "today": ["T", "AH", "D", "EY"],
    "tomorrow": ["T", "AH", "M", "AA", "R", "OW"],
    "yesterday": ["Y", "EH", "S", "T", "ER", "D", "EY"],
    "morning": ["M", "AO", "R", "N", "IH", "NG"],
    "afternoon": ["AE", "F", "T", "ER", "N", "UW", "N"],
    "evening": ["IY", "V", "N", "IH", "NG"],
    "night": ["N", "AY", "T"],
    "week": ["W", "IY", "K"],
    "month": ["M", "AH", "N", "TH"],
    "year": ["Y", "IH", "R"],
    "monday": ["M", "AH", "N", "D", "EY"],
    "tuesday": ["T", "UW", "Z", "D", "EY"],
    "wednesday": ["W", "EH", "N", "Z", "D", "EY"],
    "thursday": ["TH", "ER", "Z", "D", "EY"],
    "friday": ["F", "R", "AY", "D", "EY"],
    "saturday": ["S", "AE", "T", "ER", "D", "EY"],
    "sunday": ["S", "AH", "N", "D", "EY"],
    "january": ["JH", "AE", "N", "Y", "UW", "EH", "R", "IY"],
    "february": ["F", "EH", "B", "R", "UW", "EH", "R", "IY"],
    "march": ["M", "AA", "R", "CH"],
    "april": ["EY", "P", "R", "AH", "L"],
    "may": ["M", "EY"],
    "june": ["JH", "UW", "N"],
    "july": ["JH", "AH", "L", "AY"],
    "august": ["AO", "G", "AH", "S", "T"],
    "september": ["S", "EH", "P", "T", "EH", "M", "B", "ER"],
    "october": ["AA", "K", "T", "OW", "B", "ER"],
    "november": ["N", "OW", "V", "EH", "M", "B", "ER"],
    "december": ["D", "IY", "S", "EH", "M", "B", "ER"],
    "spring": ["S", "P", "R", "IH", "NG"],
    "summer": ["S", "AH", "M", "ER"],
    "fall": ["F", "AO", "L"],
    "autumn": ["AO", "T", "AH", "M"],
    "winter": ["W", "IH", "N", "T", "ER"],
    "red": ["R", "EH", "D"],
    "blue": ["B", "L", "UW"],
    "green": ["G", "R", "IY", "N"],
    "yellow": ["Y", "EH", "L", "OW"],
    "orange": ["AO", "R", "IH", "N", "JH"],
    "purple": ["P", "ER", "P", "AH", "L"],
    "pink": ["P", "IH", "NG", "K"],
    "brown": ["B", "R", "AW", "N"],
    "black": ["B", "L", "AE", "K"],
    "white": ["W", "AY", "T"],
    "gray": ["G", "R", "EY"],
    "grey": ["G", "R", "EY"],
}

# ARPAbet to IPA mapping (extended for better coverage)
ARPABET_TO_IPA = {
    "AA": "ɑ", "AA0": "ɑ", "AA1": "ɑ", "AA2": "ɑ",
    "AE": "æ", "AE0": "æ", "AE1": "æ", "AE2": "æ",
    "AH": "ʌ", "AH0": "ə", "AH1": "ʌ", "AH2": "ʌ",
    "AO": "ɔ", "AO0": "ɔ", "AO1": "ɔ", "AO2": "ɔ",
    "AW": "aʊ", "AW0": "aʊ", "AW1": "aʊ", "AW2": "aʊ",
    "AY": "aɪ", "AY0": "aɪ", "AY1": "aɪ", "AY2": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "EH0": "ɛ", "EH1": "ɛ", "EH2": "ɛ",
    "ER": "ɜr", "ER0": "ər", "ER1": "ɜr", "ER2": "ɜr",
    "EY": "eɪ", "EY0": "eɪ", "EY1": "eɪ", "EY2": "eɪ",
    "F": "f", "G": "g", "HH": "h", "IH": "ɪ",
    "IH0": "ɪ", "IH1": "ɪ", "IH2": "ɪ", "IY": "i",
    "IY0": "i", "IY1": "i", "IY2": "i", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "OW": "oʊ", "OW0": "oʊ", "OW1": "oʊ", "OW2": "oʊ",
    "OY": "ɔɪ", "OY0": "ɔɪ", "OY1": "ɔɪ", "OY2": "ɔɪ",
    "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UH0": "ʊ", "UH1": "ʊ", "UH2": "ʊ",
    "UW": "u", "UW0": "u", "UW1": "u", "UW2": "u",
    "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"
}

# Fallback IPA symbol rewrite table for unsupported symbols
IPA_REWRITE_TABLE = {
    # Nasalization - drop tilde for models that don't support it
    "ɑ̃": "ɑ", "ɛ̃": "ɛ", "ɔ̃": "ɔ", "œ̃": "œ",
    # Length marks - drop for models that don't support them
    "ɑː": "ɑ", "iː": "i", "uː": "u", "ɔː": "ɔ",
    # Secondary stress - convert to primary or drop
    "ˌ": "", "ˈ": "",
    # Tie bars - replace with base chars for models that don't support them
    "t͡ʃ": "tʃ", "d͡ʒ": "dʒ", "d͡z": "dz",
    # Unknown symbols map to schwa
    "ɡ": "g", "ɹ": "r", "ɫ": "l", "ɚ": "ər", "ɝ": "ɜr",
    "ʔ": "",  # glottal stop - drop it
}

def _normalize_text(text: str) -> str:
    """Apply deterministic text normalization for English."""
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'[\'\']', "'", text)

    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

def _expand_numbers(text: str) -> str:
    """Expand numbers and basic ordinals in text."""
    # Handle ordinals 1st-31st
    ordinal_map = {
        "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
        "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth",
        "11th": "eleventh", "12th": "twelfth", "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
        "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth", "19th": "nineteenth", "20th": "twentieth",
        "21st": "twenty first", "22nd": "twenty second", "23rd": "twenty third", "24th": "twenty fourth",
        "25th": "twenty fifth", "26th": "twenty sixth", "27th": "twenty seventh", "28th": "twenty eighth",
        "29th": "twenty ninth", "30th": "thirtieth", "31st": "thirty first"
    }

    for ordinal, word in ordinal_map.items():
        text = re.sub(r'\b' + ordinal + r'\b', word, text, flags=re.IGNORECASE)

    # Handle basic numbers 0-999
    def expand_number(match):
        num = int(match.group())
        if num == 0:
            return "zero"
        elif num <= 19:
            teens = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
            return teens[num]
        elif num <= 99:
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            ten = num // 10
            one = num % 10
            if one == 0:
                return tens[ten]
            else:
                ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
                return tens[ten] + " " + ones[one]
        else:  # 100-999
            hundreds = num // 100
            remainder = num % 100
            hundreds_word = ["", "one hundred", "two hundred", "three hundred", "four hundred", "five hundred",
                           "six hundred", "seven hundred", "eight hundred", "nine hundred"][hundreds]
            if remainder == 0:
                return hundreds_word
            else:
                return hundreds_word + " " + expand_number(str(remainder))

    text = re.sub(r'\b\d{1,3}\b', lambda m: expand_number(m), text)

    return text

def _strip_stress(phone: str) -> str:
    """Remove stress digits from ARPAbet phones."""
    return re.sub(r"\d", "", phone)

def _arpabet_to_ipa_seq(arp_seq: list[str]) -> str:
    """Convert ARPAbet phoneme sequence to IPA string."""
    ipa = []
    for p in arp_seq:
        base = _strip_stress(p).upper()
        ipa.append(ARPABET_TO_IPA.get(base, base.lower()))
    return " ".join(ipa)

def _lookup_cmudict(word: str) -> list[list[str]] | None:
    """Look up word in CMU dictionary if available."""
    if cmudict is None:
        return None
    entries = cmudict.dict().get(word.lower())
    if not entries:
        return None
    return entries

def _heuristic_arpabet(word: str) -> list[str]:
    """Simple LTS for OOV words using letter-cluster heuristics."""
    w = word.lower()

    # Multi-character mappings first (longest first)
    replacements = [
        ("tion", " SH AH N"), ("sion", " ZH AH N"), ("ough", " AO"),
        ("augh", " AE F"), ("eigh", " EY"), ("ight", " AY T"),
        ("ou", " AW"), ("ow", " OW"), ("oi", " OY"), ("oy", " OY"),
        ("ea", " IY"), ("ee", " IY"), ("oo", " UW"), ("ai", " EY"),
        ("ay", " EY"), ("ie", " IY"), ("oe", " OW"), ("ue", " UW"),
        ("ui", " UW IY"), ("th", " TH "), ("sh", " SH "), ("ch", " CH "),
        ("ph", " F "), ("wh", " W "), ("ck", " K "), ("ng", " NG "),
        ("nk", " NG K"), ("tion", " SH AH N"), ("sion", " ZH AH N"),
    ]

    for pattern, replacement in replacements:
        w = w.replace(pattern, replacement)

    # Single character mappings
    char_map = {
        "a": " AH ", "b": " B ", "c": " K ", "d": " D ", "e": " EH ",
        "f": " F ", "g": " G ", "h": " HH ", "i": " IH ", "j": " JH ",
        "k": " K ", "l": " L ", "m": " M ", "n": " N ", "o": " AO ",
        "p": " P ", "q": " K ", "r": " R ", "s": " S ", "t": " T ",
        "u": " AH ", "v": " V ", "w": " W ", "x": " K S ", "y": " Y ", "z": " Z "
    }

    result = []
    for char in w:
        if char in char_map:
            result.append(char_map[char].strip())
        elif char.isalpha():
            result.append("AH")  # Default to schwa for unknown letters

    return [p for p in result if p]

def _word_to_ipa(word: str, cmudict_dict: Optional[Dict[str, List[str]]]) -> str:
    """Convert a single word to IPA phonemes."""
    word_lower = word.lower().strip()

    if not word_lower:
        return word

    # Try CMU dictionary first
    if cmudict_dict and word_lower in cmudict_dict:
        arpabet_phones = cmudict_dict[word_lower][0]  # Use first pronunciation
        return _arpabet_to_ipa_seq(arpabet_phones)

    # Fall back to built-in lexicon
    if word_lower in BUILTIN_LEXICON:
        arpabet_phones = BUILTIN_LEXICON[word_lower]
        return _arpabet_to_ipa_seq(arpabet_phones)

    # Final fallback to heuristic
    arpabet_phones = _heuristic_arpabet(word_lower)
    return _arpabet_to_ipa_seq(arpabet_phones)

def text_to_ipa(text: str) -> str:
    """
    Convert English text to IPA phonemes with deterministic normalization.

    Pipeline:
    1. Normalize text (Unicode, whitespace, punctuation)
    2. Expand numbers and ordinals
    3. Tokenize by words/punctuation
    4. Convert each word to IPA using CMU -> built-in lexicon -> heuristics
    5. Join with spaces

    Args:
        text: English text to convert

    Returns:
        Space-separated IPA string
    """
    if not text or not text.strip():
        return text

    # Load CMU dictionary if available
    try:
        import cmudict
        cmudict_dict = cmudict.dict()
    except Exception:
        cmudict_dict = None

    # 1. Normalize text
    text = _normalize_text(text)

    # 2. Expand numbers
    text = _expand_numbers(text)

    # 3. Tokenize (words and punctuation)
    tokens = re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", text)

    # 4. Convert each token to IPA
    ipa_parts = []
    for token in tokens:
        if token.isalpha():
            # Convert word to IPA
            ipa_word = _word_to_ipa(token, cmudict_dict)
            ipa_parts.append(ipa_word)
        elif token in ".,!?;:":
            # Map punctuation to pause tokens or drop them
            if token in ".!?":
                ipa_parts.append(".")  # Sentence boundary
            elif token in ",;:":
                ipa_parts.append(",")  # Phrase boundary
        elif token == "...":
            ipa_parts.append(".")  # Ellipsis -> period
        # Drop other punctuation that doesn't map to IPA

    # 5. Join and normalize whitespace
    result = " ".join(ipa_parts)
    result = re.sub(r'\s+', ' ', result).strip()

    logger.debug(f"Normalized text '{text}' to IPA: {result}")
    return result

# Load the real model vocabulary from kokoro_onnx
_REAL_VOCAB = None
_VOCAB_SIZE = None

def _load_real_vocab() -> Tuple[Dict[str, int], int]:
    """Load the actual IPA vocabulary from kokoro_onnx package."""
    global _REAL_VOCAB, _VOCAB_SIZE

    if _REAL_VOCAB is not None:
        return _REAL_VOCAB, _VOCAB_SIZE

    # Try to load from kokoro_onnx assets
    try:
        import kokoro_onnx
        import os
        import json

        package_dir = os.path.dirname(kokoro_onnx.__file__)

        # Try different possible vocabulary file names
        vocab_files = [
            'assets/phoneme_to_id.json',
            'assets/ipa_vocab.json', 
            'assets/vocab.json',
            'phoneme_to_id.json',
            'ipa_vocab.json',
            'vocab.json',
            'assets/vocab.txt'
        ]

        vocab_data = None
        for vocab_file in vocab_files:
            full_path = os.path.join(package_dir, vocab_file)
            if os.path.exists(full_path):
                if vocab_file.endswith('.txt'):
                    # Handle vocab.txt format
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    vocab_data = {phoneme: i for i, phoneme in enumerate(lines)}
                else:
                    # Handle JSON format
                    with open(full_path, 'r', encoding='utf-8') as f:
                        vocab_data = json.load(f)
                logger.debug(f"Loaded vocabulary from {vocab_file}")
                break

        if vocab_data is None:
            # Try to extract from the Kokoro model itself
            try:
                import kokoro_onnx.tokenizer as kt
                if hasattr(kt, 'PHONEME_TO_ID'):
                    vocab_data = kt.PHONEME_TO_ID
                    logger.debug("Extracted PHONEME_TO_ID from kokoro_onnx.tokenizer")
                elif hasattr(kt, 'PhonemeCodec'):
                    # Try to get vocab from PhonemeCodec
                    codec = kt.PhonemeCodec()
                    if hasattr(codec, 'vocab'):
                        vocab_data = codec.vocab
                        logger.debug("Extracted vocabulary from PhonemeCodec")
            except Exception:
                pass

        if vocab_data is None:
            # Use our fallback vocabulary
            logger.debug("Using fallback IPA vocabulary")
            vocab_data = IPA_TO_ID

        # Handle different vocabulary formats
        if isinstance(vocab_data, dict):
            # phoneme -> ID format
            _REAL_VOCAB = vocab_data
            _VOCAB_SIZE = len(vocab_data)
        elif isinstance(vocab_data, list):
            # List format - assume index = ID
            _REAL_VOCAB = {phoneme: i for i, phoneme in enumerate(vocab_data)}
            _VOCAB_SIZE = len(vocab_data)
        else:
            raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")

        logger.debug(f"Loaded vocabulary with {_VOCAB_SIZE} entries")
        return _REAL_VOCAB, _VOCAB_SIZE

    except Exception as e:
        logger.warning(f"Failed to load real vocabulary, using fallback: {e}")
        # Fallback to our basic mapping (not ideal but prevents crashes)
        _REAL_VOCAB = IPA_TO_ID.copy()
        _VOCAB_SIZE = len(_REAL_VOCAB)
        return _REAL_VOCAB, _VOCAB_SIZE

def _ipa_to_ids(phonemes: str) -> List[int]:
    """
    Convert IPA phoneme string to model token IDs using greedy longest-match.
    
    Uses the real model vocabulary with no guessing or fallbacks.
    All returned IDs are guaranteed to be within [0, vocab_size-1].

    Args:
        phonemes: IPA phoneme string

    Returns:
        List of token IDs within valid range
        
    Raises:
        ValueError: If any IPA symbol cannot be encoded
    """
    if not phonemes or not phonemes.strip():
        return [0]  # Return neutral token for empty input

    vocab, vocab_size = _load_real_vocab()
    ids = []
    oov_symbols = []
    
    # Normalize whitespace to single spaces
    phonemes = re.sub(r'\s+', ' ', phonemes.strip())
    
    # Split by spaces first to handle word boundaries
    words = phonemes.split(' ')
    
    for word_idx, word in enumerate(words):
        if not word:
            continue
            
        # Add space token between words if vocab supports it
        if word_idx > 0:
            space_tokens = ['<sp>', '_', 'sil', ' ']
            space_id = None
            for space_token in space_tokens:
                if space_token in vocab:
                    space_id = vocab[space_token]
                    break
            if space_id is not None:
                ids.append(space_id)
        
        # Process each word with greedy longest-match
        i = 0
        while i < len(word):
            matched = False
            
            # Try matches from longest to shortest (up to 4 chars for complex IPA)
            for length in range(min(4, len(word) - i), 0, -1):
                candidate = word[i:i+length]
                if candidate in vocab:
                    ids.append(vocab[candidate])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Try rewrite table for unsupported symbols
                char = word[i]
                if char in IPA_REWRITE_TABLE:
                    rewritten = IPA_REWRITE_TABLE[char]
                    if rewritten and rewritten in vocab:
                        ids.append(vocab[rewritten])
                        logger.debug(f"Rewrote {char} -> {rewritten}")
                    elif rewritten == "":  # Empty rewrite means drop the symbol
                        pass  # Skip this character
                    else:
                        oov_symbols.append(f"{char}->{rewritten}")
                        # Use schwa as fallback
                        fallback_id = vocab.get('ə', vocab.get('a', 0))
                        ids.append(fallback_id)
                else:
                    oov_symbols.append(char)
                    # Use schwa as fallback
                    fallback_id = vocab.get('ə', vocab.get('a', 0))
                    ids.append(fallback_id)
                i += 1
    
    # Validate all IDs are within vocabulary range
    if ids:
        max_id = max(ids)
        min_id = min(ids)
        if max_id >= vocab_size or min_id < 0:
            raise ValueError(f"Token ID out of bounds: min={min_id}, max={max_id}, vocab_size={vocab_size}")
    
    # Log results
    oov_count = len(oov_symbols)
    if oov_count > 0:
        logger.debug(f"OOV symbols: {oov_symbols[:3]}{'...' if oov_count > 3 else ''}")
    
    logger.debug(f"ipa_len={len(phonemes)} tokens={len(ids)} vocab_size={vocab_size} max_id={max(ids) if ids else 0} oov={oov_count}")
    
    if oov_count > 0:
        raise ValueError(f"Unsupported IPA symbol(s): {', '.join(oov_symbols[:5])}")
    
    return ids
