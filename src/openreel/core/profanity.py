"""Profanity censoring for captions."""

from __future__ import annotations

import re

# Common profanity words (extend as needed)
_PROFANITY_WORDS = {
    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "fuckers",
    "fuck's",
    "shit",
    "shitty",
    "shitting",
    "bullshit",
    "bitch",
    "bitches",
    "bitching",
    "asshole",
    "assholes",
    "ass",
    "damn",
    "damned",
    "damnit",
    "dammit",
    "bastard",
    "bastards",
    "cunt",
    "cunts",
    "dick",
    "dicks",
    "dickhead",
    "piss",
    "pissed",
    "pissing",
    "cock",
    "cocks",
    "pussy",
    "whore",
    "whores",
    "slut",
    "sluts",
    "twat",
    "wanker",
    "wankers",
    "douche",
    "douchebag",
    "motherfucker",
    "motherfuckers",
    "motherfucking",
}


def _censor_word(word: str) -> str:
    """Censor a single word, preserving the first letter and length.

    e.g., 'fuck' -> 'f***', 'shit' -> 's***', 'bullshit' -> 'bullshit' (no, censored fully)
    """
    if len(word) <= 2:
        return "*" * len(word)
    return word[0] + "*" * (len(word) - 1)


def censor_text(word: str) -> str:
    """Return censored version if the word is profanity, otherwise unchanged.

    Strips punctuation for matching but preserves it in the output.
    Case-insensitive matching, case-preserving output.
    """
    # Extract the alphabetic stem (without trailing punctuation)
    match = re.match(r"^([\w']+)(.*)$", word)
    if not match:
        return word

    stem, suffix = match.group(1), match.group(2)
    lower = stem.lower()

    if lower in _PROFANITY_WORDS:
        censored = _censor_word(stem)
        # Preserve original case of first letter
        if stem[0].isupper():
            censored = censored[0].upper() + censored[1:]
        return censored + suffix

    return word
