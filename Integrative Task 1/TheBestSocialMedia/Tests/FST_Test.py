import DFA
from FST import WordCensor
import pytest

@pytest.fixture
def default_fst():
    wd = WordCensor([])
    wd.set_replace_words(DFA.get_hate_offensive())
    return wd

@pytest.fixture
def custom_fst():
    wc = WordCensor([])
    wc.set_replace_words(["spoon", "lunch", "head", "soccer"])
    return wc



def test_dfa_fuck(default_fst):
    assert default_fst.censor_words("fuck") == "f***"

def test_dfa_safe_sentence(default_fst):
    assert default_fst.censor_words("hello how are you") == "hello how are you"

def test_dfa_spam_money(default_fst):
    assert default_fst.censor_words("spam bitcoin and money money money") == "spam bitcoin and money money money"

def test_dfa_mixed_offensive(default_fst):
    assert default_fst.censor_words("There was hoe a book lesbo about elefants.") == "there was h** a book l**** about elefants"

def test_custom_fuck_unchanged(custom_fst):
    assert custom_fst.censor_words("fuck") == "fuck"

def test_custom_safe_sentence(custom_fst):
    assert custom_fst.censor_words("hello how are you") == "hello how are you"

def test_custom_spoon_lunch(custom_fst):
    assert custom_fst.censor_words("I had lunch with a spoon after my soccer match") == "i had l**** with a s**** after my s***** match"

def test_custom_head_only(custom_fst):
    assert custom_fst.censor_words("head ass helicopter kike") == "h*** ass helicopter kike"

def test_custom_punctuation_apostrophe(custom_fst):
    assert custom_fst.censor_words("we're the people that rule the world") == "were the people that rule the world"

def test_custom_punctuation_inside_word(custom_fst):
    assert custom_fst.censor_words("spo'on is censored") == "s**** is censored"

def test_custom_punctuation_mixed(custom_fst):
    assert custom_fst.censor_words("other he!ad stuff that should still be considered !soccer") == "other h*** stuff that should still be considered s*****"