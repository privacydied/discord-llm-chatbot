"""Pattern constants for curated memory curation — sensitivity, denial, classification."""

import re

# ---------------------------------------------------------------------------
# Secret patterns — blocks memories that look like API keys, tokens, etc.
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    r"\bpassword\b",
    r"\bpassphrase\b",
    r"\bapi[-_ ]?key\b",
    r"\bsecret\b",
    r"\btoken\b",
    r"\bauthorization\b",
    r"\bbearer\b",
    r"\bprivate key\b",
    r"-----BEGIN [A-Z ]+-----",
    r"\bghp_[A-Za-z0-9]{20,}\b",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
]

# ---------------------------------------------------------------------------
# Internal/tool patterns — blocks internal prompt leakage from becoming memory.
# ---------------------------------------------------------------------------
_INTERNAL_PATTERNS = [
    r"chain of thought",
    r"think step by step",
    r"internal prompt",
    r"tool trace",
    r"function call",
    r"hidden reasoning",
    r"system prompt",
]

# ---------------------------------------------------------------------------
# Project-fact anchor terms — indicates the text is about the bot/repo.
# ---------------------------------------------------------------------------
_PROJECT_FACT_SIGNALS = [
    r"\bdiscord-bot\b",
    r"\bdiscord bot\b",
    r"\bthis bot\b",
    r"\bthe bot\b",
    r"\brouter\b",
    r"\bmodule\b",
    r"\bfile\b",
    r"\brepo\b",
    r"\bimplementation\b",
    r"\bsubsystem\b",
    r"\bmemory service\b",
    r"\bmemory\b",
    r"\brag\b",
    r"\bchromadb\b",
    r"\bstt\b",
    r"\btts\b",
    r"\bvision\b",
    r"\bsearch\b",
    r"\bcommand\b",
    r"\bconfig\b",
    r"\bdeployment\b",
    r"\bruntime\b",
    r"\barchitecture\b",
    r"\baudit\b",
    r"\bbug\b",
    r"\bregression\b",
    r"\blatency\b",
    r"\bperformance\b",
    r"\breply\b",
    r"\broute\b",
    r"\btyping indicator\b",
]

# ---------------------------------------------------------------------------
# Project-fact cue verbs — "should", "uses", "needs", etc.
# ---------------------------------------------------------------------------
_PROJECT_FACT_CUES = [
    r"\bshould\b",
    r"\bmust\b",
    r"\buses\b",
    r"\bis\b",
    r"\bare\b",
    r"\bneeds?\b",
    r"\brequires?\b",
    r"\bpersists?\b",
    r"\bwraps?\b",
    r"\brejects?\b",
    r"\benables?\b",
    r"\bdisables?\b",
    r"\bdeletes?\b",
    r"\bloads?\b",
    r"\bsaves?\b",
]

# ---------------------------------------------------------------------------
# Sensitive-attribute patterns — identity, health, demographics.
# ---------------------------------------------------------------------------
_SENSITIVE_ATTRIBUTE_PATTERNS = [
    r"\brace\b",
    r"\bracial\b",
    r"\bethnic(?:ity)?\b",
    r"\bnationalit(?:y|ies)\b",
    r"\breligion\b",
    r"\breligious\b",
    r"\bpolitic(?:al|s)?\b",
    r"\bsex(?:ual|uality| orientation)?\b",
    r"\bgender\b",
    r"\btrans(?:gender)?\b",
    r"\bgay\b",
    r"\blesbian\b",
    r"\bbisexual\b",
    r"\bqueer\b",
    r"\bstraight\b",
    r"\b(?:black|white|asian|hispanic|latino)(?:\s+(?:person|people|man|woman|men|women|community|race))\b",
    r"\b(?:i|i'm|im|my)\s+(?:am\s+)?(?:black|white|asian|hispanic|latino)\b",
    r"\bjew(?:ish)?\b",
    r"\bmuslim\b",
    r"\bchristian\b",
    r"\bmale\b",
    r"\bfemale\b",
    r"\bman\b",
    r"\bwoman\b",
    r"\bmen\b",
    r"\bwomen\b",
    r"\bdisabilit(?:y|ies)\b",
    r"\bdisabled\b",
    r"\bautis(?:m|tic)\b",
    r"\badhd\b",
    r"\bdepression\b",
    r"\bdepressed\b",
    r"\bocd\b",
    r"\bbipolar\b",
]

# ---------------------------------------------------------------------------
# World-claim domain terms.
# ---------------------------------------------------------------------------
_WORLD_CLAIM_TERMS = [
    r"\bfinance\b",
    r"\bfinancial\b",
    r"\bmedia\b",
    r"\bacademia\b",
    r"\bacademic(?:s)?\b",
    r"\bsociet(?:y|al)\b",
    r"\bdating\b",
    r"\bidentity\b",
    r"\bculture\b",
    r"\bpolitic(?:s|al)?\b",
    r"\bcorporate\b",
]

# ---------------------------------------------------------------------------
# General claim comparatives — "better than", "more likely", etc.
# ---------------------------------------------------------------------------
_GENERAL_CLAIM_TERMS = [
    r"\boverrepresented\b",
    r"\bunderrepresented\b",
    r"\bdiscriminat(?:e|es|ed|ing|ion)\b",
    r"\bharmful\b",
    r"\btoxic\b",
    r"\bsuperior\b",
    r"\binferior\b",
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bmore likely\b",
    r"\bless likely\b",
    r"\bmakes up\b",
    r"\baccounts for\b",
]

# ---------------------------------------------------------------------------
# Inferred-memory denylist — blocks casual/unsafe content from becoming
# durable memories. Does NOT affect explicit !memory-add / "remember that".
# ---------------------------------------------------------------------------
_INFERRED_DENYLIST = [
    # Drugs / recreational substances / medications
    r"\b(?:xanax|xanny|xanex)\b",
    r"\b(?:coke|crack|blow|snow)\b",
    r"\b(?:heroin|bop|smack|scag)\b",
    r"\b(?:meth|mdma|molly|ecstasy|tina)\b",
    r"\b(?:lsd|acid)\b",
    r"\b(?:ketamine|special k)\b",
    r"\b(?:weed|marijuana|cannabis|bong|joint|dabs)\b",
    r"\b(?:opioid|opiate|fentanyl)\b",
    r"\b(?:ativan|clonazepam|diazepam|valium|lorazepam|alprazolam)\b",
    r"\b(?:morphine|tramadol|oxycontin|percocet|adderall)\b",
    r"\b(?:shrooms|magic mushroom|psilocybin)\b",
    # Medical / mental health
    r"\b(?:depression|depressed|anxiety|ptsd|bipolar|schizophrenia)\b",
    r"\b(?:suicid|self.harm|selfharm|suicide)\b",
    # Third-party / anecdote markers
    r"\bmy (?:friend|buddy|mate|sis|bro|cousin|girlfriend|boyfriend)\b.*\b(?:got|did|was|had|took)\b",
    r"\b(?:someone|somebody) (?:said|told|did|got|was|had|took)\b",
    r"\bi heard (?:from|that)\b",
    r"\b(?:(?:he|she|they) (?:said|told|did|got|was|had|took))\b",
    r"\bmy friends\b",
    r"\bmy friend\b",
    # Sexual / explicit content
    r"\b(?:porn|xxx|hentai|onlyfans|only\s*fans)\b",
    r"\b(?:hookup|one.night(?:\s+stand)?)\b",
    r"\b(?:nude[s]?|naked|nudes)\b",
    # Slurs (common ones)
    r"\b(?:fag|faggot|nigga|nigger|tranny|retard)\b",
]

# ---------------------------------------------------------------------------
# Public-disclosure sensitive patterns — blocks memories from being
# casually repeated in normal guild chat answers (tell-me-about-myself).
# These memories remain stored for !memories-show (owner only).
# ---------------------------------------------------------------------------
_PUBLIC_DISCLOSURE_SENSITIVE_PATTERNS = [
    # Sexual / body-size claims
    r"\b(?:porn|xxx|hentai|onlyfans|only\s*fans)\b",
    r"\b(?:hookup|one.night(?:\s+stand)?)\b",
    r"\b(?:nude[s]?|naked)\b",
    r"\b(?:sexy|hot|attractive|beautiful)\s+(?:girl|boy|woman|man|body|figure)\b",
    r"\b(?:weight|bmi|overweight|underweight|obese|skinny)\b",
    r"\b(?:size|cup|bust)\s+(?:of|is|my|her|his)\b",
    r"\bmy\s+(?:body|chest|breasts?|dick|cock|penis|vagina|pussy|ass)\b",
    r"\b(?:big|small|huge|tiny|flat)\s+(?:boobs?|chest|dick|cock|penis|ass)\b",
    # Drugs / medication / substance claims
    r"\b(?:coke|cocaine|crack|heroin|meth|mdma|molly|ecstasy|lsd|acid)\b",
    r"\b(?:xanax|xanny|xanex|valium|adderall|oxycontin|percocet|tramadol)\b",
    r"\b(?:ativan|clonazepam|diazepam|alprazolam|lorazepam)\b",
    r"\b(?:morphine|fentanyl|opiate|opioid)\b",
    r"\b(?:weed|marijuana|cannabis|bong|joint|dabs|k2)\b",
    r"\b(?:shrooms|psilocybin|ketamine|special\s*k)\b",
    r"\b(?:blow|snow|bop|smack|scag|tina)\b",
    r"\b(?:drug|high|stoned|sober|addict|addiction|overdose)\b",
    # Medical / mental-health claims
    r"\b(?:depression|depressed|anxious|anxiety|ptsd|bipolar|schizophrenia)\b",
    r"\b(?:autism|autistic|adhd|ocd|bpd|disorder)\b",
    r"\b(?:cancer|diabetes|epilepsy|hiv|aids|heart\s*disease)\b",
    r"\b(?:suicid|suicide|self.harm|selfharm)\b",
    r"\b(?:therapy|therapist|psychiatrist|medication|pills)\b",
    # Protected identity claims
    r"\b(?:race|racial|ethnic|ethnicity)\b",
    r"\b(?:religion|religious|atheist|catholic|protestant)\b",
    r"\b(?:sexual\s*orientation|gay|lesbian|bisexual|queer|transgender)\b",
    r"\b(?:political|politic|republican|democrat|liberal|conservative)\b",
    # Slurs
    r"\b(?:fag|faggot|nigga|nigger|tranny|retard|spastic)\b",
    # Third-party anecdotes
    r"\bmy\s+(?:friend|buddy|mate|sis|bro|cousin|girlfriend|boyfriend)\b.*\b(?:got|did|was|had)\b",
    r"\b(?:someone|somebody)\s+(?:said|told|did|got|was|had|took)\b",
]

# Compiled regex for is_public_safe — compile once at import.
_PUBLIC_DISCLOSURE_RE = [re.compile(p, flags=re.IGNORECASE) for p in _PUBLIC_DISCLOSURE_SENSITIVE_PATTERNS]


def is_public_safe(text: str) -> bool:
    """Return False if the memory text contains content too sensitive for casual guild-chat disclosure.

    This is applied before injecting memories into normal LLM prompts.
    The memory remains stored for !memories-show (owner only).
    """
    lower = (text or "").lower()
    return not any(p.search(lower) for p in _PUBLIC_DISCLOSURE_RE)
