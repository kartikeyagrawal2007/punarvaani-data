import pytest

from pvdata.ctc import ctc_min_length, fits_ctc
from pvdata.text_norm import normalize_transcript, number_to_hindi


@pytest.mark.parametrize(
    "n,words",
    [(0, "शून्य"), (5, "पाँच"), (25, "पच्चीस"), (99, "निन्यानबे"), (100, "एक सौ"), (2025, "दो हज़ार पच्चीस"),
     (123456, "एक लाख तेईस हज़ार चार सौ छप्पन"), (10_000_000, "एक करोड़")],
)  # fmt: skip
def test_numbers_are_spelled_in_hindi(n, words):
    assert number_to_hindi(n) == words


def test_team_guide_examples():
    assert normalize_transcript("२०२५ में ५ लोग थे").text == "दो हज़ार पच्चीस में पाँच लोग थे"
    assert normalize_transcript("2025 में 5 लोग थे").text == "दो हज़ार पच्चीस में पाँच लोग थे"
    r = normalize_transcript("School bus आई")
    assert not r.ok and r.reason == "contains_latin"


def test_punctuation_removed_danda_kept():
    assert normalize_transcript('यह, "अच्छा" है? ठीक है।').text == "यह अच्छा है ठीक है।"
    assert normalize_transcript("।। नमस्ते॥").text == "नमस्ते।"


def test_empty_after_cleaning_is_rejected():
    r = normalize_transcript("!!! ???")
    assert not r.ok and r.reason == "empty_after_cleaning"


def test_ctc_rule_matches_trainer():
    assert ctc_min_length(["A", "B", "B", "|", "C"]) == 6  # U=5, one repeat
    ok, need = fits_ctc("नमस्ते भारत", 32)
    assert ok and need < 32
    ok, need = fits_ctc("यह एक बहुत लंबा वाक्य है जिसमें बहुत सारे शब्द और ध्वनियाँ हैं", 32)
    assert not ok and need > 32
