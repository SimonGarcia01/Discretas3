from pyformlang.finite_automaton import DeterministicFiniteAutomaton, State
import re

dfa = DeterministicFiniteAutomaton()

#States
start = State("start")
safe = State("safe")
violation = State("violation")
review = State("needs review")
maybe = State("maybe needs review")

#Start State
dfa.add_start_state(start)

#Final states
dfa.add_final_state(safe)
dfa.add_final_state(violation)
dfa.add_final_state(review)

#Groups of words
hate_words = ["nigger", "kike", "lesbo"]
spam_words = ["money", "rich", "bitcoin", "wordhashtagtoken", "wordurltoken"]
offensive_words = ["fuck", "bitch", "ass", "hoe", "cunt"]


def get_hate_offensive():
    return  hate_words+offensive_words


#Hate words transitions
for word in hate_words:
    dfa.add_transition(start,word,violation)
    dfa.add_transition(maybe,word,violation)
    dfa.add_transition(review,word,violation)
    dfa.add_transition(safe,word,violation)
    dfa.add_transition(violation,word,violation)

#Offensive words transition
for word in offensive_words:
    dfa.add_transition(start,word,violation)
    dfa.add_transition(maybe,word,violation)
    dfa.add_transition(review,word,violation)
    dfa.add_transition(safe,word,violation)
    dfa.add_transition(violation,word,violation)

#Spam words transitions
for word in spam_words:
    dfa.add_transition(start,word,maybe)
    dfa.add_transition(maybe,word,review)
    dfa.add_transition(review,word,violation)
    dfa.add_transition(violation,word,violation)
    dfa.add_transition(safe,word,maybe)

#Safe words transition
safe_word = "safe"

dfa.add_transitions([
    (start, safe_word, safe),
    (safe, safe_word, safe),
    (maybe, safe_word, maybe),
    (review, safe_word, review),
    (violation, safe_word, violation),
])

def map_word(check):
    if  check in hate_words or check in offensive_words or check in spam_words:
        return check
    else:
        return safe_word

def classify_text(original_text):
    url_text = re.sub(r'https?://\w+\.\w+/*', "wordurltoken", original_text)
    url_hashtag_text = re.sub(r'#\w+', "wordhashtagtoken", url_text)
    textlist = url_hashtag_text.split()

    #DFA processing part
    current_state = start
    symbol = ""
    classification = ""

    for text in textlist:
        symbol = map_word(text)
        next_states = dfa(current_state, symbol)
        if not next_states:
            break
        current_state = next_states.pop()

    # We put conditions for distinct types of verdicts
    if  current_state == "safe":
        classification = "The post is good; there are only safe words in the post. Congratulations, you are respecting the social network policy!"

    elif current_state == "violation":
        classification = "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."
        '''
        if symbol in hate_words:
            classification = "Hate words in the post."
        elif symbol in offensive_words:
            classification = "Offensive words in the post."
        elif symbol in spam_words:
            classification = "There are an excess of spam words."
        else:
        '''
    if current_state == "maybe needs review":
        classification = "The post has safe words but it contains one spam word."
        '''
        if symbol in spam_words:
            classification = "One spam word in the post."
        elif symbol in hate_words:
            classification = "Hate words in the post."
        elif symbol in offensive_words:
            classification = "Spam words in the post."
        else:
            classification = "The post has safe words but also has spam words."
        '''
    if current_state == "needs review":
        classification = "The post must be reviewed; there are two spam words."
        '''
        if symbol in spam_words:
            classification = "The post must be reviewed; there are two spam words."
        elif symbol in hate_words:
            classification = "Hate words in the post."
        elif symbol in offensive_words:
            classification = "Offensive words in the post."
        else:
            classification = "The post is not good; there are spam two spam words in the post."
        '''

    return current_state.value, classification