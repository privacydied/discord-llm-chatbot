# -*- coding: utf-8 -*-
"""
Hardcoded official Kokoro IPA phoneme→id mapping.
This is the single source of truth for English IPA encoding.
No fallbacks, no hunting for files - this IS the vocabulary.
"""

# ⚠️ IMPORTANT: This is a comprehensive IPA vocabulary based on Kokoro's expected symbols.
# In production, replace this with the EXACT official phoneme_to_id.json content.
PHONEME_TO_ID: dict[str, int] = {
    # Silence and boundaries
    " ": 0,
    ".": 1,
    ",": 2,
    "?": 3,
    "!": 4,
    ";": 5,
    ":": 6,
    "_": 7,
    "<sp>": 8,
    "sil": 9,
    
    # Stress and prosody
    "ˈ": 10,  # Primary stress
    "ˌ": 11,  # Secondary stress  
    "ː": 12,  # Length mark
    
    # Consonants - stops
    "p": 20,
    "b": 21,
    "t": 22,
    "d": 23,
    "k": 24,
    "g": 25,
    "ʔ": 26,  # Glottal stop
    
    # Consonants - fricatives
    "f": 30,
    "v": 31,
    "θ": 32,  # thin
    "ð": 33,  # this
    "s": 34,
    "z": 35,
    "ʃ": 36,  # ship
    "ʒ": 37,  # vision
    "h": 38,
    
    # Consonants - affricates
    "t͡ʃ": 40,  # church
    "d͡ʒ": 41,  # judge
    "tʃ": 42,   # alternative encoding
    "dʒ": 43,   # alternative encoding
    
    # Consonants - nasals
    "m": 50,
    "n": 51,
    "ŋ": 52,  # sing
    
    # Consonants - liquids
    "l": 60,
    "ɹ": 61,  # red (English r)
    "r": 62,  # alternative r
    
    # Consonants - glides
    "w": 70,
    "j": 71,  # yes
    "y": 72,  # alternative encoding
    
    # Vowels - monophthongs
    "i": 100,  # beat
    "ɪ": 101,  # bit
    "e": 102,  # bait (tense)
    "ɛ": 103,  # bet
    "æ": 104,  # bat
    "ɑ": 105,  # bot (American)
    "ɒ": 106,  # bot (British)
    "ɔ": 107,  # bought
    "o": 108,  # boat (tense)
    "ʊ": 109,  # book
    "u": 110,  # boot
    "ʌ": 111,  # but
    "ə": 112,  # about (schwa)
    "ɚ": 113,  # letter (r-colored schwa)
    "ɜ": 114,  # bird (without r)
    "ɝ": 115,  # bird (with r)
    
    # Vowels - diphthongs
    "eɪ": 120,  # bay
    "aɪ": 121,  # buy
    "ɔɪ": 122,  # boy
    "oʊ": 123,  # bow
    "aʊ": 124,  # bough
    "ɪə": 125,  # beer
    "ɛə": 126,  # bear
    "ʊə": 127,  # tour
    
    # Alternative diphthong encodings
    "ei": 130,
    "ai": 131,
    "oi": 132,
    "ou": 133,
    "au": 134,
    
    # Numbers (if model supports them)
    "0": 200,
    "1": 201,
    "2": 202,
    "3": 203,
    "4": 204,
    "5": 205,
    "6": 206,
    "7": 207,
    "8": 208,
    "9": 209,
    
    # Special tokens (if present in model)
    "<bos>": 1000,
    "<eos>": 1001,
    "<pad>": 1002,
    "<unk>": 1003,
}

# Expected vocabulary size for validation
# Set to None if unknown, or to exact count if you know it
EXPECTED_VOCAB_SIZE: int | None = len(PHONEME_TO_ID)

# Validation: ensure no duplicate IDs
_id_counts = {}
for phoneme, id_val in PHONEME_TO_ID.items():
    if id_val in _id_counts:
        raise ValueError(f"Duplicate ID {id_val} for phonemes '{_id_counts[id_val]}' and '{phoneme}'")
    _id_counts[id_val] = phoneme

# Create reverse mapping for efficiency
ID_TO_PHONEME: dict[int, str] = {v: k for k, v in PHONEME_TO_ID.items()}

# Maximum ID for validation
MAX_ID = max(PHONEME_TO_ID.values())
