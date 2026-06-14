"""Local English grapheme-to-phoneme conversion outputting IPA symbols.
Zero dependencies, no network downloads, offline-only operation.
IPA-focused for Kokoro TTS model compatibility.
"""

import json
import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path

try:
    import cmudict  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cmudict = None  # type: ignore

logger = logging.getLogger(__name__)

IPA_REWRITE_TABLE: dict[str, str] = {}

# Load lexicon from external JSON file
_LEXICON_CACHE = None

# Optional shared tokenizer instance from kokoro_onnx when available.
_OFFICIAL_TOKENIZER_STATE = "uninitialized"
_OFFICIAL_TOKENIZER = None
_ESPEAK_TMPDIR_CONFIGURED = False
_ESPEAK_TMPDIR_PATH: Path | None = None


def get_kokoro_tempdir() -> Path:
    """Return the directory used for espeak temp artifacts."""
    override = os.environ.get("KOKORO_ESPEAK_TMPDIR") or os.environ.get("KOKORO_PHONEMIZER_TMPDIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / ".kokoro_espeak_tmp"


class G2PUnavailableError(RuntimeError):
    """Raised when the deterministic G2P pipeline cannot be used."""


def _load_lexicon() -> dict[str, str]:
    """Load lexicon from lexicon_en.json file."""
    global _LEXICON_CACHE
    if _LEXICON_CACHE is not None:
        return _LEXICON_CACHE

    try:
        lexicon_path = Path(__file__).parent / "lexicon_en.json"
        if lexicon_path.exists():
            with open(lexicon_path, encoding="utf-8") as f:
                _LEXICON_CACHE = json.load(f)
                logger.debug("Loaded English lexicon: %d entries", len(_LEXICON_CACHE))
                return _LEXICON_CACHE
    except Exception as e:
        logger.warning("Failed to load lexicon_en.json: %s", e)

    _LEXICON_CACHE = {}
    return _LEXICON_CACHE


def apply_lexicon(text: str) -> str:
    """Apply lexicon replacements to text."""
    lexicon = _load_lexicon()
    if not lexicon:
        return text

    words = text.split()
    out = []
    for w in words:
        key = w.lower().strip(":;,.!?")
        if key in lexicon:
            out.append(lexicon[key])
        else:
            out.append(w)
    return " ".join(out)


def normalize_text(text: str) -> str:
    """Normalize text for TTS while preserving prosody-relevant punctuation.

    Keeps sentence and phrase boundary markers (. , ! ? ; :) for natural speech rhythm.
    Expands numbers to words for proper pronunciation.
    [REH][CA]
    """
    t = unicodedata.normalize("NFKC", text)

    # Map colons and semicolons to commas (phrase boundaries) rather than dropping
    t = t.replace(":", ", ").replace(";", ", ")

    # Keep prosody-relevant punctuation: . , ! ? ' -
    # Also keep parentheses content but remove the parens themselves
    t = re.sub(r"[()\[\]{}]", " ", t)

    # Remove remaining special chars but preserve alphanumeric and prosody marks
    t = re.sub(r"[^A-Za-z0-9'.,?!\-\s]", " ", t)

    # Normalize multiple punctuation marks
    t = re.sub(r"([.,!?])\1+", r"\1", t)  # Collapse repeated punctuation
    t = re.sub(r"\s+", " ", t).strip()

    # Handle contractions (preserve for natural speech)
    # Don't strip apostrophes from contractions like "what's"

    # Expand numbers to words for proper pronunciation
    return number_to_words(t)



def number_to_words(text: str) -> str:
    """Convert numbers to spoken words for natural TTS pronunciation.

    Handles:
    - Small numbers (0-99)
    - Years (1900-2099)
    - Large numbers with commas (1,000)
    - Ordinals (1st, 2nd, 3rd)
    [REH]
    """
    # Basic number words
    ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    def _small_num(n: int) -> str:
        """Convert 0-99 to words."""
        if n < 20:
            return ones[n]
        if n < 100:
            t, o = divmod(n, 10)
            return tens[t] + (" " + ones[o] if o else "")
        return str(n)  # Fallback for unexpected values

    def _year_to_words(year: int) -> str:
        """Convert year (1900-2099) to spoken form."""
        if 1900 <= year <= 1999:
            return f"{ones[19]} {_small_num(year - 1900)}"
        if 2000 <= year <= 2009:
            return "two thousand" + (f" {ones[year - 2000]}" if year > 2000 else "")
        if 2010 <= year <= 2099:
            return f"twenty {_small_num(year - 2000)}"
        return str(year)

    def _convert_number(s: str) -> str:
        """Convert a numeric string to words."""
        # Strip commas from large numbers
        s_clean = s.replace(",", "")

        # Check if it's a valid integer
        if not s_clean.isdigit():
            return s

        n = int(s_clean)

        # Years (4-digit, reasonable range)
        if 1900 <= n <= 2099 and len(s_clean) == 4:
            return _year_to_words(n)

        # Small numbers
        if n == 0:
            return "zero"
        if n < 100:
            return _small_num(n)
        if n < 1000:
            h, r = divmod(n, 100)
            result = ones[h] + " hundred"
            if r:
                result += " " + _small_num(r)
            return result
        if n < 10000:
            th, r = divmod(n, 1000)
            result = _small_num(th) + " thousand"
            if r >= 100:
                result += " " + _convert_number(str(r))
            elif r > 0:
                result += " " + _small_num(r)
            return result

        # For very large numbers, just return original
        return s

    # Ordinal suffixes
    ordinal_map = {
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
        "4th": "fourth",
        "5th": "fifth",
        "6th": "sixth",
        "7th": "seventh",
        "8th": "eighth",
        "9th": "ninth",
        "10th": "tenth",
        "11th": "eleventh",
        "12th": "twelfth",
        "13th": "thirteenth",
        "20th": "twentieth",
        "30th": "thirtieth",
        "21st": "twenty first",
        "22nd": "twenty second",
        "23rd": "twenty third",
    }

    words = text.split()
    out = []
    for word in words:
        # Strip trailing punctuation for matching
        punct = ""
        clean_word = word
        while clean_word and clean_word[-1] in ".,!?;:":
            punct = clean_word[-1] + punct
            clean_word = clean_word[:-1]

        # Check ordinals first
        if clean_word.lower() in ordinal_map:
            out.append(ordinal_map[clean_word.lower()] + punct)
        # Check pure numbers or comma-separated numbers
        elif re.match(r"^\d{1,3}(,\d{3})*$", clean_word) or clean_word.isdigit():
            out.append(_convert_number(clean_word) + punct)
        else:
            out.append(word)

    return " ".join(out)


# REMOVED: No fallback vocabulary allowed for English IPA synthesis.
# All vocabulary loading must go through official sources via ipa_vocab_loader.py
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
    "sentence": ["S", "EH", "N", "T", "AH", "N", "S"],
    "grammar": ["G", "R", "AE", "M", "ER"],
    "vocabulary": ["V", "OW", "K", "AE", "B", "Y", "AH", "L", "EH", "R", "IY"],
    "conversation": ["K", "AA", "N", "V", "ER", "S", "EY", "SH", "AH", "N"],
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
    "angry": ["AE", "NG", "G", "R", "IY"],
    "tired": ["T", "AY", "ER", "D"],
    "hungry": ["HH", "AH", "NG", "G", "R", "IY"],
    "thirsty": ["TH", "ER", "S", "T", "IY"],
    "sick": ["S", "IH", "K"],
    "healthy": ["HH", "EH", "L", "TH", "IY"],
    "busy": ["B", "IH", "Z", "IY"],
    "free": ["F", "R", "IY"],
    "ready": ["R", "EH", "D", "IY"],
    "late": ["L", "EY", "T"],
    "early": ["ER", "L", "IY"],
    "difficult": ["D", "IH", "F", "AH", "K", "AH", "L", "T"],
    "expensive": ["IH", "K", "S", "P", "EH", "N", "S", "IH", "V"],
    "cheap": ["CH", "IY", "P"],
    "clean": ["K", "L", "IY", "N"],
    "dirty": ["D", "ER", "T", "IY"],
    "beautiful": ["B", "Y", "UW", "T", "AH", "F", "AH", "L"],
    "ugly": ["AH", "G", "L", "IY"],
    "rich": ["R", "IH", "CH"],
    "poor": ["P", "UH", "R"],
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
    "soft": ["S", "AO", "F", "T"],
    "wet": ["W", "EH", "T"],
    "dry": ["D", "R", "AY"],
    "full": ["F", "UH", "L"],
    "empty": ["EH", "M", "P", "T", "IY"],
    "open": ["OW", "P", "AH", "N"],
    "closed": ["K", "L", "OW", "Z", "D"],
    "true": ["T", "R", "UW"],
    "false": ["F", "AO", "L", "S"],
    "real": ["R", "IY", "L"],
    "fake": ["F", "EY", "K"],
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
    "twenty seventh": [
        "T",
        "W",
        "EH1",
        "N",
        "T",
        "IY",
        "S",
        "EH1",
        "V",
        "AH",
        "N",
        "TH",
    ],
    "twenty eighth": ["T", "W", "EH1", "N", "T", "IY", "EY1", "T", "TH"],
    "twenty ninth": ["T", "W", "EH1", "N", "T", "IY", "N", "AY1", "N", "TH"],
    "thirtieth": ["TH", "ER1", "T", "IY", "TH"],
    "thirty first": ["TH", "ER1", "T", "IY", "F", "ER1", "S", "T"],
    "counting": ["K", "AW", "N", "T", "IH", "NG"],
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
    "into": ["IH", "N", "T", "UW"],
    "above": ["AH", "B", "AH", "V"],
    "below": ["B", "IH", "L", "OW"],
    "left": ["L", "EH", "F", "T"],
    "front": ["F", "R", "AH", "N", "T"],
    "back": ["B", "AE", "K"],
    "inside": ["IH", "N", "S", "AY", "D"],
    "outside": ["AW", "T", "S", "AY", "D"],
    "up": ["AH", "P"],
    "down": ["D", "AW", "N"],
    "here": ["HH", "IY", "R"],
    "there": ["DH", "EH", "R"],
    "now": ["N", "AW"],
    "today": ["T", "AH", "D", "EY"],
    "tomorrow": ["T", "AH", "M", "AA", "R", "OW"],
    "yesterday": ["Y", "EH", "S", "T", "ER", "D", "EY"],
    "morning": ["M", "AO", "R", "N", "IH", "NG"],
    "afternoon": ["AE", "F", "T", "ER", "N", "UW", "N"],
    "evening": ["IY", "V", "N", "IH", "NG"],
    "night": ["N", "AY", "T"],
    "week": ["W", "IY", "K"],
    "month": ["M", "AH", "N", "TH"],
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
    "AA": "ɑ",
    "AA0": "ɑ",
    "AA1": "ɑ",
    "AA2": "ɑ",
    "AE": "æ",
    "AE0": "æ",
    "AE1": "æ",
    "AE2": "æ",
    "AH": "ʌ",
    "AH0": "ə",
    "AH1": "ʌ",
    "AH2": "ʌ",
    "AO": "ɔ",
    "AO0": "ɔ",
    "AO1": "ɔ",
    "AO2": "ɔ",
    "AW": "aʊ",
    "AW0": "aʊ",
    "AW1": "aʊ",
    "AW2": "aʊ",
    "AY": "aɪ",
    "AY0": "aɪ",
    "AY1": "aɪ",
    "AY2": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "EH0": "ɛ",
    "EH1": "ɛ",
    "EH2": "ɛ",
    "ER": "ɚ",
    "ER0": "ɚ",
    "ER1": "ɚ",
    "ER2": "ɚ",
    "EY": "eɪ",
    "EY0": "eɪ",
    "EY1": "eɪ",
    "EY2": "eɪ",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ɪ",
    "IH0": "ɪ",
    "IH1": "ɪ",
    "IH2": "ɪ",
    "IY": "i",
    "IY0": "i",
    "IY1": "i",
    "IY2": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OW0": "oʊ",
    "OW1": "oʊ",
    "OW2": "oʊ",
    "OY": "ɔɪ",
    "OY0": "ɔɪ",
    "OY1": "ɔɪ",
    "OY2": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UH0": "ʊ",
    "UH1": "ʊ",
    "UH2": "ʊ",
    "UW": "u",
    "UW0": "u",
    "UW1": "u",
    "UW2": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

# REMOVED: No IPA rewrite fallbacks allowed for English.
# All IPA symbols must be handled by the official vocab loader.


def _normalize_text(text: str) -> str:
    """Apply deterministic text normalization for English."""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[\'\']", "'", text)

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()



def _expand_numbers(text: str) -> str:
    """Expand numbers and basic ordinals in text."""
    # Handle ordinals 1st-31st
    ordinal_map = {
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
        "4th": "fourth",
        "5th": "fifth",
        "6th": "sixth",
        "7th": "seventh",
        "8th": "eighth",
        "9th": "ninth",
        "10th": "tenth",
        "11th": "eleventh",
        "12th": "twelfth",
        "13th": "thirteenth",
        "14th": "fourteenth",
        "15th": "fifteenth",
        "16th": "sixteenth",
        "17th": "seventeenth",
        "18th": "eighteenth",
        "19th": "nineteenth",
        "20th": "twentieth",
        "21st": "twenty first",
        "22nd": "twenty second",
        "23rd": "twenty third",
        "24th": "twenty fourth",
        "25th": "twenty fifth",
        "26th": "twenty sixth",
        "27th": "twenty seventh",
        "28th": "twenty eighth",
        "29th": "twenty ninth",
        "30th": "thirtieth",
        "31st": "thirty first",
    }

    for ordinal, word in ordinal_map.items():
        text = re.sub(r"\b" + ordinal + r"\b", word, text, flags=re.IGNORECASE)

    # Handle basic numbers 0-999
    def expand_number(match):
        num = int(match.group())
        if num == 0:
            return "zero"
        if num <= 19:
            teens = [
                "",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
            return teens[num]
        if num <= 99:
            tens = [
                "",
                "",
                "twenty",
                "thirty",
                "forty",
                "fifty",
                "sixty",
                "seventy",
                "eighty",
                "ninety",
            ]
            ten = num // 10
            one = num % 10
            if one == 0:
                return tens[ten]
            ones = [
                "",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]
            return tens[ten] + " " + ones[one]
        # 100-999
        hundreds = num // 100
        remainder = num % 100
        hundreds_word = [
            "",
            "one hundred",
            "two hundred",
            "three hundred",
            "four hundred",
            "five hundred",
            "six hundred",
            "seven hundred",
            "eight hundred",
            "nine hundred",
        ][hundreds]
        if remainder == 0:
            return hundreds_word
        return hundreds_word + " " + expand_number(str(remainder))

    return re.sub(r"\b\d{1,3}\b", expand_number, text)



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
        ("tion", " SH AH N"),
        ("sion", " ZH AH N"),
        ("ough", " AO"),
        ("augh", " AE F"),
        ("eigh", " EY"),
        ("ight", " AY T"),
        ("ou", " AW"),
        ("ow", " OW"),
        ("oi", " OY"),
        ("oy", " OY"),
        ("ea", " IY"),
        ("ee", " IY"),
        ("oo", " UW"),
        ("ai", " EY"),
        ("ay", " EY"),
        ("ie", " IY"),
        ("oe", " OW"),
        ("ue", " UW"),
        ("ui", " UW IY"),
        ("th", " TH "),
        ("sh", " SH "),
        ("ch", " CH "),
        ("ph", " F "),
        ("wh", " W "),
        ("ck", " K "),
        ("ng", " NG "),
        ("nk", " NG K"),
        ("tion", " SH AH N"),
        ("sion", " ZH AH N"),
    ]

    for pattern, replacement in replacements:
        w = w.replace(pattern, replacement)

    # Single character mappings
    char_map = {
        "a": " AE ",
        "b": " B ",
        "c": " K ",
        "d": " D ",
        "e": " EH ",
        "f": " F ",
        "g": " G ",
        "h": " HH ",
        "i": " IH ",
        "j": " JH ",
        "k": " K ",
        "l": " L ",
        "m": " M ",
        "n": " N ",
        "o": " AO ",
        "p": " P ",
        "q": " K ",
        "r": " R ",
        "s": " S ",
        "t": " T ",
        "u": " UH ",
        "v": " V ",
        "w": " W ",
        "x": " K S ",
        "y": " Y ",
        "z": " Z ",
    }

    result = []
    for char in w:
        if char in char_map:
            result.append(char_map[char].strip())
        elif char.isalpha():
            result.append("AH")  # Default to schwa for unknown letters

    return [p for p in result if p]


def _word_to_ipa(word: str, cmudict_dict: dict[str, list[str]] | None) -> str:
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


def _get_official_tokenizer():
    """Return a cached kokoro_onnx Tokenizer instance when available."""
    global _OFFICIAL_TOKENIZER_STATE, _OFFICIAL_TOKENIZER

    if _OFFICIAL_TOKENIZER_STATE == "failed":
        return None
    if _OFFICIAL_TOKENIZER_STATE == "ready":
        return _OFFICIAL_TOKENIZER

    try:  # pragma: no cover - exercised via tests through monkeypatch
        from kokoro_onnx.tokenizer import Tokenizer  # type: ignore

        _OFFICIAL_TOKENIZER = Tokenizer()
        _OFFICIAL_TOKENIZER_STATE = "ready"
        logger.debug("Initialized official Kokoro tokenizer for IPA phonemization")
    except Exception:  # pragma: no cover - dependency may be missing
        _OFFICIAL_TOKENIZER = None
        _OFFICIAL_TOKENIZER_STATE = "failed"
        logger.debug(
            "Official Kokoro tokenizer unavailable; falling back to CMU pipeline",
            exc_info=True,
        )
    return _OFFICIAL_TOKENIZER


def _should_retry_official_tokenizer(exc: Exception) -> bool:
    """Return True when the failure looks like an espeak tempdir issue."""
    message = str(exc).lower()
    retry_signatures = (
        "failed to map segment from shared object",
        "cannot allocate memory in static tls block",
    )
    if any(signature in message for signature in retry_signatures):
        return True

    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, Exception) and _should_retry_official_tokenizer(cause):
        return True

    context = getattr(exc, "__context__", None)
    return bool(isinstance(context, Exception) and _should_retry_official_tokenizer(context))


def _configure_official_tokenizer_tmpdir() -> Path | None:
    """Route espeak temp files to an executable-safe location."""
    global _ESPEAK_TMPDIR_CONFIGURED, _ESPEAK_TMPDIR_PATH

    if _ESPEAK_TMPDIR_CONFIGURED:
        return _ESPEAK_TMPDIR_PATH

    candidate = get_kokoro_tempdir()

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except Exception as dir_error:  # pragma: no cover - defensive
        logger.warning(
            "Unable to prepare espeak temporary directory %s: %s",
            candidate,
            dir_error,
        )
        _ESPEAK_TMPDIR_CONFIGURED = True
        _ESPEAK_TMPDIR_PATH = None
        return None

    tempfile.tempdir = str(candidate)
    os.environ["TMPDIR"] = str(candidate)
    os.environ["TEMP"] = str(candidate)
    os.environ["TMP"] = str(candidate)

    _ESPEAK_TMPDIR_CONFIGURED = True
    _ESPEAK_TMPDIR_PATH = candidate
    logger.debug("Configured Kokoro espeak temp directory: %s", candidate)
    return candidate


def _phonemize_with_official(tokenizer, text: str) -> str:
    ipa = tokenizer.phonemize(text, lang="en-us", norm=True)
    return " ".join(str(ipa).split())


def _attempt_official_tokenizer(tokenizer, text: str) -> str | None:
    """Try the official tokenizer with a safe retry for tempdir failures."""
    for attempt in range(2):
        try:
            ipa = _phonemize_with_official(tokenizer, text)
        except Exception as exc:  # pragma: no cover - depends on runtime setup
            logger.debug(
                "Official Kokoro tokenizer attempt %d for '%s' failed: %s",
                attempt + 1,
                text,
                exc,
            )
            should_retry = attempt == 0 and _should_retry_official_tokenizer(exc)
            if should_retry:
                tmpdir = _configure_official_tokenizer_tmpdir()
                if tmpdir is not None:
                    logger.info(
                        "Official Kokoro tokenizer retrying with safe temp directory %s",
                        tmpdir,
                    )
                    continue
            logger.warning(
                "Official Kokoro tokenizer failed; reverting to CMU pipeline",
                exc_info=True,
            )
            return None

        if ipa:
            logger.debug(
                "Official Kokoro tokenizer converted '%s' to IPA: %s",
                text,
                ipa,
            )
            return ipa

        logger.debug(
            "Official Kokoro tokenizer returned empty IPA for '%s'; falling back.",
            text,
        )
        return None

    return None


def text_to_ipa(text: str) -> str:
    """Convert English text to IPA phonemes with deterministic normalization.

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
    [REH][CA]

    """
    if not text or not text.strip():
        return text

    raw_text = text  # Keep original for logging

    tokenizer = _get_official_tokenizer()
    if tokenizer is not None:
        ipa = _attempt_official_tokenizer(tokenizer, text)
        if ipa:
            logger.debug(
                "g2p.official chars_in=%d ipa_len=%d preview=%s",
                len(raw_text),
                len(ipa),
                repr(ipa[:50]) if len(ipa) > 50 else repr(ipa),
                extra={
                    "subsys": "g2p",
                    "event": "official_tokenizer",
                    "chars": len(raw_text),
                },
            )
            return ipa

    # Load CMU dictionary if available
    try:
        import cmudict

        cmudict_dict = cmudict.dict()
    except Exception as exc:
        msg = "CMU Pronouncing Dictionary is not available for English IPA synthesis"
        raise G2PUnavailableError(msg) from exc

    # 1. Normalize text and apply lexicon
    text = normalize_text(text)
    text_after_lexicon = apply_lexicon(text)

    # Log normalization step for debug visibility [REH]
    if text != text_after_lexicon:
        logger.debug(
            "g2p.lexicon before=%s after=%s",
            repr(text[:40]),
            repr(text_after_lexicon[:40]),
            extra={"subsys": "g2p", "event": "lexicon_applied"},
        )
    text = text_after_lexicon

    # 3. Tokenize (words and punctuation)
    tokens = re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", text)

    # 4. Convert each token to IPA
    ipa_parts = []
    punct_count = 0
    word_count = 0

    for token in tokens:
        if token.isalpha():
            # Convert word to IPA
            ipa_word = _word_to_ipa(token, cmudict_dict)
            ipa_parts.append(ipa_word)
            word_count += 1
        elif token in ".,!?;:":  # nosec B105
            # Map punctuation to pause tokens - critical for prosody! [CA]
            if token in ".!?":  # nosec B105
                ipa_parts.append(".")  # Sentence boundary
                punct_count += 1
            elif token in ",;:":  # nosec B105
                ipa_parts.append(",")  # Phrase boundary
                punct_count += 1
        elif token == "...":  # nosec B105
            ipa_parts.append(".")  # Ellipsis -> period
            punct_count += 1
        # Drop other punctuation that doesn't map to IPA

    # 5. Join and normalize whitespace
    result = " ".join(ipa_parts)
    result = re.sub(r"\s+", " ", result).strip()

    # Structured logging for pipeline visibility [REH]
    logger.debug(
        "g2p.complete chars_in=%d words=%d punct=%d ipa_len=%d preview=%s",
        len(raw_text),
        word_count,
        punct_count,
        len(result),
        repr(result[:60]) if len(result) > 60 else repr(result),
        extra={
            "subsys": "g2p",
            "event": "g2p_complete",
            "chars_in": len(raw_text),
            "words": word_count,
            "punct": punct_count,
            "ipa_len": len(result),
        },
    )
    return result


# Load the real model vocabulary from kokoro_onnx
_REAL_VOCAB = None
_VOCAB_SIZE = None


def _load_real_vocab():
    """Load the official hardcoded Kokoro IPA vocabulary.
    No external files or fallbacks - uses the embedded mapping.
    """
    global _REAL_VOCAB, _VOCAB_SIZE

    if _REAL_VOCAB is not None and _VOCAB_SIZE is not None:
        return _REAL_VOCAB, _VOCAB_SIZE

    try:
        # Import from the hardcoded vocabulary
        from bot.tts.ipa_vocab_kokoro_v1 import PHONEME_TO_ID

        _REAL_VOCAB = dict(PHONEME_TO_ID)
        _VOCAB_SIZE = len(_REAL_VOCAB)

        logger.debug(f"Loaded hardcoded vocabulary with {_VOCAB_SIZE} entries")
        return _REAL_VOCAB, _VOCAB_SIZE

    except Exception as e:
        msg = f"Failed to load hardcoded Kokoro IPA vocabulary: {e}. No fallbacks allowed for English."
        raise RuntimeError(msg)


def _ipa_to_ids(phonemes: str) -> list[int]:
    """Convert IPA phoneme string to model token IDs using greedy longest-match.

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
    phonemes = re.sub(r"\s+", " ", phonemes.strip())

    # Split by spaces first to handle word boundaries
    words = phonemes.split(" ")

    for word_idx, word in enumerate(words):
        if not word:
            continue

        # Add space token between words if vocab supports it
        if word_idx > 0:
            space_tokens = ["<sp>", "_", "sil", " "]
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
                candidate = word[i : i + length]
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
                        fallback_id = vocab.get("ə", vocab.get("a", 0))
                        ids.append(fallback_id)
                else:
                    oov_symbols.append(char)
                    # Use schwa as fallback
                    fallback_id = vocab.get("ə", vocab.get("a", 0))
                    ids.append(fallback_id)
                i += 1

    # Validate all IDs are within vocabulary range
    if ids:
        max_id = max(ids)
        min_id = min(ids)
        if max_id >= vocab_size or min_id < 0:
            msg = f"Token ID out of bounds: min={min_id}, max={max_id}, vocab_size={vocab_size}"
            raise ValueError(msg)

    # Log results
    oov_count = len(oov_symbols)
    if oov_count > 0:
        logger.debug(f"OOV symbols: {oov_symbols[:3]}{'...' if oov_count > 3 else ''}")

    logger.debug(f"ipa_len={len(phonemes)} tokens={len(ids)} vocab_size={vocab_size} max_id={max(ids) if ids else 0} oov={oov_count}")

    if oov_count > 0:
        msg = f"Unsupported IPA symbol(s): {', '.join(oov_symbols[:5])}"
        raise ValueError(msg)

    return ids
