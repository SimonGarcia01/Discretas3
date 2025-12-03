from pyformlang.fst import FST
import DFA
import re

class WordCensor:
    #offensive words must be in a list (["sum","other-sum",...])
    def __init__(self, offensive_words):
        self.words_to_replace = offensive_words
        self.transducer = FST()
        self.build_transducer()

    def build_transducer(self):
        global transducer
        transducer = FST()
        for word in self.words_to_replace:
            transducer.add_transition("s0", word, "s1", [word[0] + '*'*(len(word)-1)])
        transducer.add_start_state("s0")
        transducer.add_final_state("s1")

    def censor_words(self, text):
        text_no_punctuation = re.sub(r'[^\w\s]', "", text)
        textlist = text_no_punctuation.lower().strip().split()
        censored_tokens = []

        for w in textlist:
            if w in self.words_to_replace:
                outputs = list(transducer.translate([w]))
                results = ["".join(o) for o in outputs]
                censored_tokens.append(results[0])
            else:
                censored_tokens.append(w)

        return " ".join(censored_tokens)

    def set_replace_words(self,list_of_words):
        self.words_to_replace = list_of_words
        self.build_transducer()




