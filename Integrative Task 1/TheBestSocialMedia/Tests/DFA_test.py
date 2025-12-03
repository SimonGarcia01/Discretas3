import DFA


def test_map_word():
    assert DFA.map_word("Anything") == "safe"
    assert DFA.map_word("money") == "money"
    assert DFA.map_word("zzz") == "safe"
    assert DFA.map_word("cunt") == "cunt"
    assert DFA.map_word("https://example.com") == "safe"
    assert DFA.map_word("kike") == "kike"


def test_safe_word_only():
    assert DFA.classify_text("safe words only")[1] == "The post is good; there are only safe words in the post. Congratulations, you are respecting the social network policy!"

def test_one_spam_word():
    assert DFA.classify_text("money")[1] == "The post has safe words but it contains one spam word."

def test_two_spam_words():
    assert DFA.classify_text("get very rich with bitcoin")[1] == "The post must be reviewed; there are two spam words."

def test_too_much_spam_words():
    assert DFA.classify_text("get very rich with bitcoin earn tons of money")[1] == "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."

def test_one_link_spam():
    assert DFA.classify_text("try here at https://example.com")[1] == "The post has safe words but it contains one spam word."


def test_link_hashtag_spam():
    assert DFA.classify_text("general link https://example.com and a hashtag #live")[1] == "The post must be reviewed; there are two spam words."

def test_link_followed_by_hashtag():
    assert DFA.classify_text("mixing https://google.com #hashtag and bitcoin")[1] == "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."

def test_one_bad_word():
    assert DFA.classify_text("fuck")[1] == "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."

def test_safe_before_bad_word():
    assert DFA.classify_text("a bunch of safe words until bitch")[1] == "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."

def test_violation_between_safe():
    assert DFA.classify_text("great happy kike words")[1] == "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."

def test_bad_words_and_hashtag():
    assert DFA.classify_text("happy #spam fuck nigger")[1] == "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."


