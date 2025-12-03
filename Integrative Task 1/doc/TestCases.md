# DFA Scenario Configurations

| Name | Class| Scenario                          |
|------|------|-----------------------------------|
| None | DFA  |  The DFA is imported normally.  |

---

# DFA Test Cases Design

**Test’s objective:** *Test the DFA method to return the considered hate speech and offensive words* 

| Class        | Method              | Scenario | Input | Expected Output |
|--------------|---------------------|----------|-------|-----------------|
|DFA| get_hate_offensive() | None   | N/A   | A List containing the words "nigger", "kike", "lesbo", "fuck", "bitch", "ass", "hoe" and "cunt"|

---

**Test’s objective:** *Verify the classification of words taking the lists of words into consideration.* 

| Class        | Method              | Scenario | Input | Expected Output |
|--------------|---------------------|----------|-------|-----------------|
|DFA| map_word() | None   | "Anything"  | "safe" |
|DFA| map_word() | None   | "money"  | "money" |
|DFA| map_word() | None   | "zzz"  | "safe" |
|DFA| map_word() | None   | "cunt"  | "cunt" |
|DFA| map_word() | None   | "https://example.com"  | "safe" |
|DFA| map_word() | None   | "kike"  | "kike" |
---

**Test’s objective:** *Test that DFA returns the correct classification of entire texts.* 

| Class        | Method              | Scenario | Input | Expected Output |
|--------------|---------------------|----------|-------|-----------------|
|DFA| classify_text() | None   | "safe words only"  | "The post is good; there are only safe words in the post. Congratulations, you are respecting the social network policy!" |
|DFA| classify_text() | None   | "money"  | "The post has safe words but it contains one spam word."  |
|DFA| classify_text() | None   | "get very rich with bitcoin"  | "The post must be reviewed; there are two spam words."  |
|DFA| classify_text() | None   | "get very rich with bitcoin earn tons of money"  | "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."  |
|DFA| classify_text() | None   | "try here at https://example.com"  | "The post has safe words but it contains one spam word."  |
|DFA| classify_text() | None   | "general link https://example.com  and a hashtag #live"  | "The post must be reviewed; there are two spam words."  |
|DFA| classify_text() | None   | "mixing https://google.com #hashtag and bitcoin"  | "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."  |
|DFA| classify_text() | None   | "fuck"  | "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."  |
|DFA| classify_text() | None   | "a bunch of safe words until bitch"  | "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."  |
|DFA| classify_text() | None   | "great happy kike words"  | "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."  |
|DFA| classify_text() | None   | "happy #spam fuck nigger"  | "It doesn't matter if there are safe words in the post, you have committed a violation of the social network policy."  |
---

# FST Scenario Configurations

| Name | Class| Scenario                          |
|------|------|-----------------------------------|
| None | FST |  The FST is imported normally.  |
| custom setup | FST | The FST is initialized and given the words "spoon", "lunch", "head" and "soccer" to store as the words to replace (uses set_replace_words()).

---

# FST Test Cases Design

**Test’s objective:** *Test the offesive and hate speech words taken from the DFA.* 

| Class        | Method              | Scenario | Input | Expected Output |
|--------------|---------------------|----------|-------|-----------------|
|FST| censor_words() | None   | "fuck" | "f***" |
|FST| censor_words() | None   | "hello how are you" | "hello how are you" |
|FST| censor_words() | None   | "spam bitcoin and money money money" | "spam bitcoin and money money money" |
|FST| censor_words() | None   | "There was hoe a book lesbo about elefants." | "There was h** a book l**** about elefants." |
---

**Test’s objective:** *Test that the FST will censor anything that is added to the list of censored words.* 

| Class        | Method              | Scenario | Input | Expected Output |
|--------------|---------------------|----------|-------|-----------------|
|FST| censor_words() | custom setup   | "fuck" | "fuck" |
|FST| censor_words() | custom setup   | "hello how are you" | "hello how are you" |
|FST| censor_words() | custom setup  | "I had lunch with a spoon after my soccer match" | "I had l**** with a s**** after my soccer match" |
|FST| censor_words() | custom setup  | "head ass helicopter kike" | "h*** ass helicopter kike" |
---

**Test’s objective:** *Test punctuation after the FST censoring.* 

| Class        | Method              | Scenario | Input | Expected Output |
|--------------|---------------------|----------|-------|-----------------|
|FST| censor_words() | custom setup   | "we're the people that rule the world" | "were the people that rule the world" |
|FST| censor_words() | custom setup   | "spo'on is censored" | "s**** is censored" |
|FST| censor_words() | custom setup  | "other he!ad stuff that should still be considered !soccer" | "other h*** stuff that should still be considered s*****" |
---

# CFG Scenario Configurations

| Name | Class| Scenario                          |
|------|------|-----------------------------------|
| None | CFG  | The CFG is imported. |

---

# CFG Test Cases Design

**Test’s objective:** *Test every production within the CFG alone.*  

| Class | Method | Scenario | Input | Expected Output |
|-------|---------|----------|-------|-----------------|
| CFG | `text_to_html()` | None | `"only raw text"` | ```<p>only raw text</p>``` |
| CFG | `text_to_html()` | None | `"*bold*"` | ```<p><b>bold</b></p>``` |
| CFG | `text_to_html()` | None | `"-italic-"` | ```<p><i>italic</i></p>``` |
| CFG | `text_to_html()` | None | `"{underlined}"` | ```<p><u>underlined</u></p>``` |
| CFG | `text_to_html()` | None | `"~crossed~"` | ```<p><s>bold</s></p>``` |
| CFG | `text_to_html()` | None | `":)"` | ```<p>&#128578;</p>``` |
| CFG | `text_to_html()` | None | `"#topic"` | ```<p><span class="hashtag">#topic</span></p>``` |
| CFG | `text_to_html()` | None | `"@user123"` | ```<p><span class="mention">@user123</span></p>``` |
| CFG | `text_to_html()` | None | `"<https://example.com>"` | ```<p><a href="https://example.com">https://example.com</a></p>``` |
| CFG | `text_to_html()` | None | `"$2$"` | ```<p><span class="formula">2</span></p>``` |
| CFG | `text_to_html()` | None | `"$x+3=z$"` | ```<p><span class="formula">x+3=z</span></p>``` |
| CFG | `text_to_html()` | None | `"$x_(2)$"` | ```<p><span class="formula"><sub>2</sub></span></p>``` |
| CFG | `text_to_html()` | None | `"$x^(3)$"` | ```<p><span class="formula"><sup>3</sup></span></p>``` |
| CFG | `text_to_html()` | None | `"$sin(2)$"` | ```<p><span class="formula">sin(2)</span></p>``` |
| CFG | `text_to_html()` | None | `"#hashtag *bold* -italic- {underline} ~strike~ @user <https://xd.com>"` | ```<p><span class="hashtag">#hashtag</span><b>bold</b><i>italic</i><u>underline</u><s>strike</s><span class="mention">@user</span><a href="https://xd.com">https://xd.com</a></p>``` |
| CFG | `text_to_html()` | None | `"$y = a_(n) * x^(2) + b - 5$"` | ```<p><span class="formula"> y a<sub>n</sub>x<sup>2</sup></p>``` |
| CFG | `text_to_html()` | None | `"$z_(i) + w_(j) - k_(10)$"` | ```<p><span class="formula">z<sub>i</sub>w<sub>j</sub>k<sub>10</sub></p>``` |
| CFG | `text_to_html()` | None | `"$x^(2) + y^(10) - m^(k)$"` | ```<p><span class="formula">x<sup>2</sup>y<sup>10</sup>m<sup>k</sup></p>``` |
| CFG | `text_to_html()` | None | `"$a_(n)^(2) + b_(m)^(3) - c_(x)^(y)$"` | ```<p><span class="formula">a<sub>n</sub><sup>2</sup>b<sub>m</sub><sup>3</sup>c<sub>x</sub><sup>y</sup></p>``` |
| CFG | `text_to_html()` | None | `"$sin(x) + cos(y) - f(x)$"` | ```<p>sin(x)+cos(y)-f(x)</p>``` |
| CFG | `text_to_html()` | None | `"%%%$$$@@@###$"` | ```<span class='error'>Syntax error:``` |

---

**Test’s objective:** *Test mixed posts with various elements to parse.*  

| Class | Method | Scenario | Input | Expected Output |
|-------|---------|----------|-------|-----------------|
| CFG | `text_to_html()` | None | `"Hello *world* :D"` | ```<p>Hello <b>world</b> &#128515;</p>``` |
| CFG | `text_to_html()` | None | `"Check this #topic @user"` | ```<p>Check this <span class="hashtag">#topic</span><span class="mention">@user</span></p>``` |
| CFG | `text_to_html()` | None | `"Visit <https://test.com/12d_#4> and solve $x+1$"` | ```<p>Visit <a href="https://test.com/12d_#4">https://test.com/12d_#4</a>and solve <span class="formula">x+1</span></p>``` |
| CFG | `text_to_html()` | None | ```Hey @science_fam :D I just read an amazing article https://physicsworld.com\n about motion.\nCheck this important -note- {here} ~carefully~ :o\n\nThe formula we use is:\n$ F_(net) = m * a + c * v_(1) - k * x^(2) + f(t) / m - 9 + h^(2) - g * m $\n\nWhat do you think? #physics #math``` | ```<p>Hey <span class="mention">@science_fam</span>&#128515;I just read an amazing article https://physicsworld.com\n about motion.\nCheck this important <i>note</i><u>here</u><s>carefully</s>&#128558;The formula we use is:\n<span class="formula">F<sub>net</sub>=m*a+c*v<sub>1</sub>-k*x<sup>2</sup>+f(t)/m-9+h<sup>2</sup>-g*m</span>What do you think? <span class="hashtag">#physics</span><span class="hashtag">#math</span></p>``` |

---