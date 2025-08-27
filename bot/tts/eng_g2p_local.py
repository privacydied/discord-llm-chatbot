from __future__ import annotations
import re
try:
    # cmudict carries the full CMU pronouncing dictionary without NLTK downloads
    import cmudict  # type: ignore
except Exception:  # pragma: no cover
    cmudict = None  # we handle OOV / no-package fallback below

# Core ARPAbet -> IPA map (extend as needed)
ARPABET_TO_IPA = {
    "AA":"ɑ", "AE":"æ", "AH":"ʌ", "AO":"ɔ", "AW":"aʊ", "AY":"aɪ",
    "B":"b", "CH":"t͡ʃ", "D":"d", "DH":"ð",
    "EH":"ɛ", "ER":"ɝ", "EY":"eɪ", "F":"f", "G":"ɡ", "HH":"h",
    "IH":"ɪ", "IY":"i", "JH":"d͡ʒ", "K":"k", "L":"l", "M":"m",
    "N":"n", "NG":"ŋ", "OW":"oʊ", "OY":"ɔɪ", "P":"p", "R":"ɹ",
    "S":"s", "SH":"ʃ", "T":"t", "TH":"θ", "UH":"ʊ", "UW":"u",
    "V":"v", "W":"w", "Y":"j", "Z":"z", "ZH":"ʒ"
}

_WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")

def _strip_stress(phone: str) -> str:
    # Remove stress digits (e.g., AH0 -> AH)
    return re.sub(r"\d", "", phone)

def _arpabet_to_ipa_seq(arp_seq: list[str]) -> str:
    ipa = []
    for p in arp_seq:
        base = _strip_stress(p).upper()
        ipa.append(ARPABET_TO_IPA.get(base, base.lower()))
    return " ".join(ipa)

def _lookup_cmudict(word: str) -> list[list[str]] | None:
    if cmudict is None:
        return None
    entries = cmudict.dict().get(word.lower())
    if not entries:
        return None
    return entries  # list of ARPAbet lists

def _heuristic_arpabet(word: str) -> list[str]:
    # Ultra-simple LTS for OOV words (keeps us phoneme-based, never graphemes)
    w = word.lower()
    # Common digraphs first
    w = w.replace("ch", " CH ").replace("sh", " SH ").replace("th", " TH ").replace("ng", " NG ")
    # Vowel heuristics
    w = re.sub(r"[aeiou]y\b", " IY", w)
    w = re.sub(r"[aeiou]+", " AH ", w)
    # Consonant heuristics
    repl = {
        "b":" B ","c":" K ","d":" D ","f":" F ","g":" G ","h":" HH ","j":" JH ","k":" K ","l":" L ","m":" M ",
        "n":" N ","p":" P ","q":" K ","r":" R ","s":" S ","t":" T ","v":" V ","w":" W ","x":" K S ","y":" Y ","z":" Z "
    }
    for k,v in repl.items(): w = w.replace(k, v)
    phones = re.findall(r"[A-Z]{1,2}", w)
    return phones or ["AH"]  # at least a vowel

def text_to_ipa(text: str) -> str:
    tokens = _WORD_RE.findall(text)
    ipa_out: list[str] = []
    for tok in tokens:
        if tok.isalpha():
            cmu = _lookup_cmudict(tok)
            if cmu:
                # choose first pronunciation
                ipa_out.append(_arpabet_to_ipa_seq(cmu[0]))
            else:
                ipa_out.append(_arpabet_to_ipa_seq(_heuristic_arpabet(tok)))
        else:
            # keep punctuation/nums as short pause markers
            ipa_out.append("")
    return " ".join(filter(None, ipa_out))

# IPA symbol to model ID mapping (centered around zero, within [-178, 177])
IPA_TO_ID = {
    # Vowels
    "ɑ": 0, "æ": 1, "ʌ": 2, "ɔ": 3, "ə": 4, "ɜ": 5, "ɛ": 6, "ɪ": 7, "i": 8,
    "o": 9, "ʊ": 10, "u": 11, "a": 12, "e": 13,

    # Diphthongs
    "aʊ": 14, "aɪ": 15, "eɪ": 16, "oʊ": 17, "ɔɪ": 18,

    # Consonants
    "b": 19, "d": 20, "f": 21, "g": 22, "h": 23, "j": 24, "k": 25, "l": 26,
    "m": 27, "n": 28, "p": 29, "r": 30, "s": 31, "t": 32, "v": 33, "w": 34,
    "z": 35, "ð": 36, "θ": 37, "ʃ": 38, "ʒ": 39, "tʃ": 40, "dʒ": 41, "ŋ": 42,

    # Length and stress markers (map to silence/zero)
    "ː": 0, "ˈ": 0, "ˌ": 0, "̃": 0,
}

# Valid ID range for the model
IPA_VOCAB_HALF = 178
VALID_ID_MIN = -IPA_VOCAB_HALF
VALID_ID_MAX = IPA_VOCAB_HALF - 1

def _ipa_to_ids(phonemes: str) -> list[int]:
    """
    Convert IPA phoneme string to model token IDs within valid range [-178, 177].

    Args:
        phonemes: IPA phoneme string

    Returns:
        List of token IDs within valid range
    """
    if not phonemes or not phonemes.strip():
        return [0]  # Return neutral token for empty input

    ids = []
    sanitized_count = 0

    # Process each character, prioritizing multi-char units
    i = 0
    while i < len(phonemes):
        # Try 3-char matches first (e.g., "tʃ", "dʒ")
        if i + 2 < len(phonemes):
            three_char = phonemes[i:i+3]
            if three_char in IPA_TO_ID:
                ids.append(IPA_TO_ID[three_char])
                i += 3
                continue

        # Try 2-char matches (e.g., "aʊ", "aɪ")
        if i + 1 < len(phonemes):
            two_char = phonemes[i:i+2]
            if two_char in IPA_TO_ID:
                ids.append(IPA_TO_ID[two_char])
                i += 2
                continue

        # Single character match
        one_char = phonemes[i]
        if one_char in IPA_TO_ID:
            ids.append(IPA_TO_ID[one_char])
        else:
            # Unknown symbol maps to neutral (0)
            ids.append(0)
            sanitized_count += 1

        i += 1

    # Sanitize all IDs to valid range
    for j in range(len(ids)):
        if ids[j] < VALID_ID_MIN or ids[j] > VALID_ID_MAX:
            sanitized_count += 1
            # Clip to valid range
            ids[j] = max(VALID_ID_MIN, min(VALID_ID_MAX, ids[j]))

    # Log sanitization if any occurred
    if sanitized_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"sanitized_ipa_ids={sanitized_count}")

    return ids
