"""Transcript cleaning to the TEAM_GUIDE rules.

1. Pure Devanagari: Latin letters are not allowed (the clip is rejected unless already transliterated).
2. Numbers are spelled out in Hindi words: "2025" -> "दो हज़ार पच्चीस", "५" -> "पाँच".
3. Punctuation is removed; only spaces and the danda (।) are kept.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# 0-99 are irregular in Hindi, so they are listed in full.
_UNITS = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानबे बानबे तिरानबे चौरानबे पंचानबे छियानबे सत्तानबे अट्ठानबे निन्यानबे"
).split()
assert len(_UNITS) == 100

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_LATIN = re.compile(r"[A-Za-z]")
_DEVANAGARI_WORD = re.compile(r"^[ऀ-ॿ]+$")


def number_to_hindi(n: int) -> str:
    """Spell a non-negative integer in Hindi words (Indian system: सौ, हज़ार, लाख, करोड़)."""
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n < 100:
        return _UNITS[n]
    parts: list[str] = []
    for value, name in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            q, n = divmod(n, value)
            parts.append(f"{number_to_hindi(q)} {name}")
    if n:
        parts.append(_UNITS[n])
    return " ".join(parts)


def _spell_numbers(text: str) -> str:
    text = text.translate(_DEVANAGARI_DIGITS)
    text = re.sub(r"(?<=\d)[,](?=\d)", "", text)  # 1,000 -> 1000

    def repl(m: re.Match) -> str:
        whole, frac = m.group(1), m.group(2)
        words = number_to_hindi(int(whole))
        if frac:  # 2.5 -> दो दशमलव पाँच
            words += " दशमलव " + " ".join(_UNITS[int(d)] for d in frac)
        return f" {words} "

    text = re.sub(r"(\d+)(?:\.(\d+))?", repl, text)
    text = text.replace("%", " प्रतिशत ")
    return text


@dataclass
class NormResult:
    text: str
    ok: bool
    reason: str = ""


def normalize_transcript(raw: str, spell_out_numbers: bool = True, reject_latin: bool = True) -> NormResult:
    """Clean an ASR transcript to the TEAM_GUIDE format, or say why it can't be used."""
    text = unicodedata.normalize("NFC", raw)
    if spell_out_numbers:
        text = _spell_numbers(text)
    elif re.search(r"[0-9०-९]", text):
        return NormResult("", False, "contains_digits")
    if _LATIN.search(text):
        if reject_latin:
            return NormResult("", False, "contains_latin")
        text = _LATIN.sub(" ", text)
    text = text.replace("॥", "।")
    out_words = []
    for token in text.replace("।", " । ").split():
        if token == "।":
            out_words.append(token)
            continue
        cleaned = "".join(ch for ch in token if "ऀ" <= ch <= "ॿ" and ch not in "।॥")
        if cleaned:
            out_words.append(cleaned)
    # no danda at the start, and no repeated dandas
    while out_words and out_words[0] == "।":
        out_words.pop(0)
    dedup: list[str] = []
    for w in out_words:
        if w == "।" and dedup and dedup[-1] == "।":
            continue
        dedup.append(w)
    clean = " ".join(dedup).replace(" ।", "।").strip()
    if not any(_DEVANAGARI_WORD.match(w) for w in dedup if w != "।"):
        return NormResult("", False, "empty_after_cleaning")
    return NormResult(clean, True)
