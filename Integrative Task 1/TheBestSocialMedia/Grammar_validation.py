import os
from textx import metamodel_from_file, TextXSyntaxError

grammar = os.path.join(os.path.dirname(__file__), "Grammar.tx")


def parse_and_render(text):
    try:
        metamodel = metamodel_from_file(grammar)
        syntax_tree = metamodel.model_from_str(text)
        html = text_to_html(syntax_tree)
        # Put everything within a paragraph
        return f"<p>{html}</p>"
    except TextXSyntaxError as e:
        return f"<span class='error'>Syntax error: {str(e)}</span>"
    except Exception as e:
        return f"<span class='error'>Internal error: {str(e)}</span>"

#For the emojis in Unicode
EMOJI_MAP = {
    ":)": "&#128578;",
    ":(": "&#128577;",
    ":D": "&#128515;",
    ":p": "&#128539;",
    ":o": "&#128558;",
    ":":": ",
}


def text_to_html(model):
    html = ""

    for elem in model.elements:

        # Links
        if elem.link:
            html += f'<a href="{elem.link.value}">{elem.link.value}</a> '

        # Hashtags
        elif elem.hashtag:
            html += f'<span class="hashtag">#{elem.hashtag.value}</span> '

        # Mentions
        elif elem.mention:
            html += f'<span class="mention">@{elem.mention.value}</span> '

        # Formulas
        elif elem.formula:
            html += '<span class="formula">'
            for p in elem.formula.value:
                if p.operator:
                    html += p.operator.operator
                elif p.number:
                    html += str(p.number.value)
                elif p.variable:
                    html += p.variable.value
                elif p.subscript:
                    nv = p.subscript.value
                    val = nv.number.value if nv.number else nv.variable.value
                    html += f'<sub>{val}</sub> '
                elif p.superscript:
                    nv = p.superscript.value
                    val = nv.number.value if nv.number else nv.variable.value
                    html += f'<sup>{val}</sup> '
                elif p.function:
                    nv = p.function.value
                    arg = nv.number.value if nv.number else nv.variable.value
                    html += f'{p.function.name}({arg})'
            html += '</span> '

        #Text
        elif elem.text:
            for part in elem.text.content:
                if part.bold:
                    html += f'<b>{part.bold.value.value}</b> '
                elif part.italic:
                    html += f'<i>{part.italic.value.value}</i> '
                elif part.underline:
                    html += f'<u>{part.underline.value.value}</u> '
                elif part.strikethrough:
                    html += f'<s>{part.strikethrough.value.value}</s> '
                elif part.emoji:
                    html += EMOJI_MAP.get(part.emoji.value)
                elif part.rawtext:
                    html += part.rawtext.value
                elif part.whitespace:
                    html += part.whitespace.value

        # Whitespaces
        elif elem.whitespace:
            html += elem.whitespace.value

    return html