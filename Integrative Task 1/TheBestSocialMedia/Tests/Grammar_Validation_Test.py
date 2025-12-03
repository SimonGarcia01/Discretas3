import Grammar_validation


####################################################################################################################
# BASIC TESTS FOR EVERY ELEMENT IN GRAMMAR #########################################################################
####################################################################################################################

def test_raw_text():
    html = Grammar_validation.parse_and_render("only raw text")
    assert html == "<p>only raw text</p>"


def test_bold():
    html = Grammar_validation.parse_and_render("*bold*")
    assert html == "<p><b>bold</b> </p>"


def test_italic():
    html = Grammar_validation.parse_and_render("-italic-")
    assert html == "<p><i>italic</i> </p>"


def test_underlined():
    html = Grammar_validation.parse_and_render("{underlined}")
    assert html == "<p><u>underlined</u> </p>"


def test_crossed():
    html = Grammar_validation.parse_and_render("~crossed~")
    assert html == "<p><s>crossed</s> </p>"


def test_emoji_smile():
    html = Grammar_validation.parse_and_render(":)")
    assert html == "<p>&#128578;</p>"


def test_hashtag():
    html = Grammar_validation.parse_and_render("#topic")
    assert html == '<p><span class="hashtag">#topic</span> </p>'


def test_mention():
    html = Grammar_validation.parse_and_render("@user123")
    assert html == '<p><span class="mention">@user123</span> </p>'


def test_link():
    html = Grammar_validation.parse_and_render("<https://example.com>")
    assert html == '<p><a href="https://example.com">https://example.com</a> </p>'


def test_formula_number():
    html = Grammar_validation.parse_and_render("$2$")
    assert html == '<p><span class="formula">2</span> </p>'


def test_formula_equation():
    html = Grammar_validation.parse_and_render("$x+3=z$")
    assert html == '<p><span class="formula">x+3=z</span> </p>'


def test_formula_subscript():
    html = Grammar_validation.parse_and_render("$x_(2)$")
    assert html == '<p><span class="formula">x<sub>2</sub> </span> </p>'


def test_formula_superscript():
    html = Grammar_validation.parse_and_render("$x^(3)$")
    assert html == '<p><span class="formula">x<sup>3</sup> </span> </p>'


def test_formula_function():
    html = Grammar_validation.parse_and_render("$sin(2)$")
    assert html == '<p><span class="formula">sin(2)</span> </p>'

def test_parse_basic_elements():
    input_text = "#hashtag *bold* -italic- {underline} ~strike~ @user <https://xd.com>"
    html = Grammar_validation.parse_and_render(input_text)

    assert '<span class="hashtag">#hashtag</span>' in html
    assert '<b>bold</b>' in html
    assert '<i>italic</i>' in html
    assert '<u>underline</u>' in html
    assert '<s>strike</s>' in html
    assert '<span class="mention">@user</span>' in html
    assert '<a href="https://xd.com">https://xd.com</a>' in html


def test_parse_formula1():
    input_text = "$y = a_(n) * x^(2) + b - 5$"
    html = Grammar_validation.parse_and_render(input_text)

    assert '<span class="formula">' in html
    assert 'y' in html
    assert 'a<sub>n</sub>' in html
    assert 'x<sup>2</sup>' in html

def test_parse_formula2():
    input_text = "$z_(i) + w_(j) - k_(10)$"
    html = Grammar_validation.parse_and_render(input_text)
    assert '<span class="formula">' in html
    assert 'z<sub>i</sub>' in html
    assert 'w<sub>j</sub>' in html
    assert 'k<sub>10</sub>' in html


def test_parse_formula3():
    input_text = "$x^(2) + y^(10) - m^(k)$"
    html = Grammar_validation.parse_and_render(input_text)
    assert '<span class="formula">' in html
    assert 'x<sup>2</sup>' in html
    assert 'y<sup>10</sup>' in html
    assert 'm<sup>k</sup>' in html


def test_parse_formula4():
    input_text = "$a_(n)^(2) + b_(m)^(3) - c_(x)^(y)$"
    html = Grammar_validation.parse_and_render(input_text)
    assert '<span class="formula">' in html
    assert 'a<sub>n</sub> <sup>2</sup>' in html
    assert 'b<sub>m</sub> <sup>3</sup>' in html
    assert 'c<sub>x</sub> <sup>y</sup>' in html


def test_parse_formula5():
    input_text= "$sin(x) + cos(y) - f(x)$"
    html = Grammar_validation.parse_and_render(input_text)
    assert '<span class="formula">' in html
    assert 'sin(x)+cos(y)-f(x)' in html


def test_invalid_input():
    input_text = "%%%$$$@@@###"
    html = Grammar_validation.parse_and_render(input_text)
    assert "<span class='error'>Syntax error:" in html

####################################################################################################################
# TESTS THAT USE MULTIPLE ELEMENTS OF THE GRAMMAR ##################################################################
####################################################################################################################

def test_mixed_text_emoji():
    html = Grammar_validation.parse_and_render("Hello *world* :D")
    assert html == '<p>Hello <b>world</b> &#128515;</p>'


def test_mixed_hashtag_mention():
    html = Grammar_validation.parse_and_render("Check this #topic @user")
    assert html == '<p>Check this <span class="hashtag">#topic</span> <span class="mention">@user</span> </p>'


def test_mixed_link_formula():
    html = Grammar_validation.parse_and_render("Visit <https://test.com/12d_#4> and solve $x+1$")
    assert html == '<p>Visit <a href="https://test.com/12d_#4">https://test.com/12d_#4</a> and solve <span class="formula">x+1</span> </p>'


def test_large_integration():
    input_text = """Hey @science_fam :D I just read an amazing article https://physicsworld.com
 about motion.
Check this important -note- {here} ~carefully~ :o

The formula we use is:
$ F_(net) = m * a + c * v_(1) - k * x^(2) + f(t) / m - 9 + h^(2) - g * m $

What do you think? #physics #math"""
    html = Grammar_validation.parse_and_render(input_text)

    expected = '<p>Hey <span class="mention">@science_fam</span> &#128515;I just read an amazing article https: //physicsworld.com\n about motion.\nCheck this important <i>note</i> <u>here</u> <s>carefully</s> &#128558;The formula we use is: <span class="formula">F<sub>net</sub> =m*a+c*v<sub>1</sub> -k*x<sup>2</sup> +f(t)/m-9+h<sup>2</sup> -g*m</span> What do you think? <span class="hashtag">#physics</span> <span class="hashtag">#math</span> </p>'
    assert html.strip() == expected.strip()
