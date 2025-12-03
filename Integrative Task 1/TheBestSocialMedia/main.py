# For run, open the terminal and put the following command:
# uvicorn main:app --reload
#Enjoy it!

# Dependencies
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Imports for do the implementation of the integrative task
import re

from starlette.middleware.cors import CORSMiddleware

import DFA

from FST import WordCensor
censor = WordCensor([])
censor.set_replace_words(DFA.get_hate_offensive())
import Grammar_validation


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="static/html")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('postClassification.html', {"request": request, "preview": ""})

@app.post('/clasify')
async def classifyContent(request: Request):
    data = await request.json()
    text = data.get("content", "")
    if text != "":
        classif = DFA.classify_text(text)
        censored = censor.censor_words(text)
        enchanted = Grammar_validation.parse_and_render(text)
        if "Syntax error" in enchanted: enchanted = text



        return {

            "status": "success",
            "preview": enchanted,
            "classification": classif[0] if classif else "...",
            "classification_message": classif[1] if classif else "...",
            "censored": censored,
            "regex": validateRegEx(text)


        }
    return {
        "status": "",
        "preview": "",
        "classification": "",
        "classification_message": "",
        "censored": "",
        "regex": ""
    }


def validateRegEx(text: str):

    text = re.sub(r'\s{2,}', ' ', text.strip())
    # This variable is for the text not put in lower because this put wrong the emojis, like :D
    textForEmojis = text
    text = text.lower()

    hashtag_pattern = r'#\w+'
    # ARREGLAR PATRON ANTERIOR
    # ESTE FUNCIONA BIEN
    url_pattern = r'\<(https?://)?(\w+W*\.)(\w{2,}/?)((\w*\W*\.?/?)*)\>'
    mention_pattern = r'@\w+'
    emoji_map = {
        ":)": "&#128578;",
        ":(": "&#128577;",
        ":D": "&#128515;",
        ":p": "&#128539;",
        ":o": "&#128558;",
    }

    hashtags = re.findall(hashtag_pattern, text)
    urls = re.findall(url_pattern, text)
    mentions = re.findall(mention_pattern, text)

    emojis = []


    for word in textForEmojis.split():
        for emoji_text, emoji_html in emoji_map.items():
            if emoji_text in word:
                emojis.append(emoji_html)

    urls = [''.join(url) for url in urls]

    result_html = "<ul>"

    if hashtags:
        result_html += "<li><strong>Hashtags:</strong> " + ", ".join(hashtags) + "</li>"

    if urls:
        result_html += "<li><strong>Links:</strong> " + ", ".join(urls) + "</li>"

    if mentions:
        result_html += "<li><strong>Mentions:</strong> " + ", ".join(mentions) + "</li>"

    if emojis:
        result_html += "<li><strong>Emojis:</strong> " + ", ".join(emojis) + "</li>"

    result_html += "</ul>"

    if not (hashtags or urls or mentions or emojis):
        result_html = "<p><em>No special elements found</em></p>"

    return result_html
