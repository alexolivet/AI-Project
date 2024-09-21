####
## Import libraries
####
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    got_request_exception,
    session,
    send_from_directory,
    send_file,
    url_for,
)
import config
import config_stability_ai
import config_newsapi
import openai
from openai import OpenAI

# from openai.APIError import RateLimitError,AuthenticationError,Timeout,APIError,APIConnectionError,InvalidRequestError,ServiceUnavailableError
import os
import urllib.parse as urlparse
from urllib.parse import urljoin
import os.path
import sqlite3
import datetime
import traceback
import sys
from io import BytesIO  # for decoding base64 image
from PIL import Image
import requests
from requests.exceptions import RequestException
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import pyttsx3
import re
import validators
import pathlib
import base64
import binascii
import pandas as pd
import random
import asyncio
import json
import shutil

# Initializing the engine
# engine = pyttsx3.init()
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, *sys.argv)


###### Start Ollama libraries ######
import ollama
import urllib3

# Disable SSL verification for requests library
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma

# from langchain.vectorstores.chroma import Chroma
from langchain_community import embeddings

# import inspect
# from inspect import getmembers, isfunction
from langchain_community.chat_models import ChatOllama

# print(getmembers(ChatOllama, isfunction))
# print(dir(ChatOllama))
# print(inspect.getfullargspec(ChatOllama))
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # crafts prompts for our llm

# chunking strategy
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter
import nltk

nltk.download("punkt", download_dir="/home/alexaj14/virtualenv/nltk_data")
from nltk.tokenize.punkt import PunktSentenceTokenizer
import spacy

nlp = spacy.load("en_core_web_sm")
# Document Specific Splitting - Python
from langchain.text_splitter import PythonCodeTextSplitter

from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.llms.ollama_functions import (
    OllamaFunctions,
    DEFAULT_RESPONSE_FUNCTION,
)
from langchain.schema import SystemMessage

###### Recursive Character Text Splitting ######
from langchain.text_splitter import RecursiveCharacterTextSplitter

###### chroma db ######
import chromadb
from chromadb.config import Settings

####### chat history array ##########
chat_history = []  # stores message history
####### models and vector db settings ##########
llm = Ollama(model="dolphin-phi")
model_local = ChatOllama(model="dolphin-phi")
embed_model = "mxbai-embed-large"
embedding = embeddings.OllamaEmbeddings(model=embed_model)
# `percentile` (default) — In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.
# `standard_deviation` — In this method, any difference greater than X standard deviations is split.
# `interquartile` — In this method, the interquartile distance is used to split chunks.

###### end Ollama Embedding ######
# api_key=config.DevelopmentConfig.OPENAI_KEY
# openai.api_key= api_key
IMG_FOLDER = os.path.join("static", "img_variation")
RESEARCH_IMAGE = os.path.join("static", "image_research_ai")
VIDEO_FOLDER = os.path.join("static", "video")
PDF_FOLDER = os.path.join("static", "pdf")
client = OpenAI(api_key=config.DevelopmentConfig.OPENAI_KEY)
api_key_stability = config_stability_ai.DevelopmentConfigStability.STABILITY_KEY
newsapi_key = config_newsapi.DevelopmentConfigNewsapi.NEWS_API_KEY


# testing moderation
def moderate_text(input_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DevelopmentConfig.OPENAI_KEY}",
    }
    data = {"input": input_text}
    response = requests.post(
        "https://api.openai.com/v1/moderations", headers=headers, json=data
    )
    response.raise_for_status()
    result = response.json()

    if result["results"][0]["flagged"]:
        print("Content blocked: Violates usage policies.")
        print("Categories:", result["results"][0]["categories"])
    else:
        print("Content allowed.")


# Initialize the specific client
def get_chroma_client_specific(specific_directory):
    global Chromaclient_specific
    #   Chromaclient_specific = chromadb.Client(Settings(
    #                                 #    chroma_db_impl="duckdb+parquet",
    #                                     persist_directory=f"./vectorDatabase/{specific_directory}",
    #                                      anonymized_telemetry=False, is_persistent=True,
    #                                 ))
    Chromaclient_specific = chromadb.PersistentClient(
        path=f"./vectorDatabase/{specific_directory}"
    )
    return Chromaclient_specific


def view_chroma_database_specific():
    global collections_specific
    collections_specific = (
        Chromaclient_specific.list_collections()
    )  # get the list of collections
    print("view_chroma_database", collections_specific)
    if not collections_specific:  # check if collections list is empty
        print("There are no collections")
        return collections_specific


# connect to db
def get_db_connection():
    conn = sqlite3.connect("./database/maestro.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    return conn


def page_not_found(e):
    return render_template("404.html", page="Chatbot project", **locals()), 404


app = Flask(__name__, static_url_path="/static", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["UPLOAD_FOLDER"] = IMG_FOLDER
app.config.from_object(config.config["development"])
app.register_error_handler(404, page_not_found)
# Max size of the file
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
# configuring the allowed extensions
allowed_extensions = ["jpg", "png", "pdf"]


# only accept certain file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


####
## function calling tools
####


# Function to get news articles
def get_news(query):
    """
    Fetches a latest news from an API.

    Returns:
    str: A string cor list containing a title,shot article description and a url to the news article or an error message
    """

    logging.info(f"Getting news for {query}")
    # Define the endpoint
    url = "https://newsapi.org/v2/everything?"
    # Specify the query and number of returns
    parameters = {
        "q": query,  # query phrase
        "pageSize": "1",
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": newsapi_key,  # your own API key
    }

    try:
        # Make the request
        response = requests.get(url, params=parameters)
        response.raise_for_status()
        # Check out the dictionaries keys
        print(response)

        # Convert the response to JSON format
        response_json = response.json()

        response = []
        for i in response_json["articles"]:
            fetchedNews = (
                f"Title: {i['title']} ,Description:  {i['description']},URL: {i['url']}"
            )
            response.append(fetchedNews)
        logging.info(f"News result: {response}")
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching news: {str(e)}")
        return f"An error occurred while fetching news: {str(e)}"


def get_current_weather(city, unit="celsius"):
    """
    Fetches the current weather for a given location.

    Args:
    location (str): The city and country, e.g., "San Francisco, USA"
    unit (str): Temperature unit, either "celsius" or "fahrenheit"

    Returns:
    str: A string describing the current weather, or an error message
    """
    logging.info(f"Getting weather for {city}")
    base_url = "https://api.open-meteo.com/v1/forecast"

    # Set up parameters for the weather API
    params = {
        "latitude": 0,
        "longitude": 0,
        "current_weather": "true",
        "temperature_unit": unit,
    }

    # Set up geocoding to convert location name to coordinates
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    location_parts = city.split(",")
    city = location_parts[0].strip()
    country = location_parts[1].strip() if len(location_parts) > 1 else ""

    geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}

    try:
        # First attempt to get coordinates
        logging.info(f"Fetching coordinates for {city}")
        geo_response = requests.get(geocoding_url, params=geo_params)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        logging.debug(f"Geocoding response: {geo_data}")

        # If first attempt fails, try with full location string
        if "results" not in geo_data or not geo_data["results"]:
            geo_params["name"] = city
            geo_response = requests.get(geocoding_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            logging.debug(f"Second geocoding attempt response: {geo_data}")

        # Extract coordinates if found
        if "results" in geo_data and geo_data["results"]:
            params["latitude"] = geo_data["results"][0]["latitude"]
            params["longitude"] = geo_data["results"][0]["longitude"]
            logging.info(
                f"Coordinates found: {params['latitude']}, {params['longitude']}"
            )
        else:
            logging.warning(f"No results found for location: {city}")
            return f"Sorry, I couldn't find the location: {city}"

        # Fetch weather data using coordinates
        logging.info("Fetching weather data")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        logging.debug(f"Weather data response: {weather_data}")

        # Extract and format weather information
        if "current_weather" in weather_data:
            current_weather = weather_data["current_weather"]
            temp = current_weather["temperature"]
            wind_speed = current_weather["windspeed"]

            result = f"The current weather in {city} is {temp}°{unit.upper()} with a wind speed of {wind_speed} km/h."
            logging.info(f"Weather result: {result}")
            return result
        else:
            logging.warning(f"No current weather data found for {city}")
            return f"Sorry, I couldn't retrieve weather data for {city}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching weather data: {str(e)}")
        return f"An error occurred while fetching weather data: {str(e)}"


# Function to get a random joke
def get_random_joke():
    """
    Fetches a random joke from an API.

    Returns:
    str: A string containing a joke, or an error message
    """
    logging.info("Fetching a random joke")
    joke_url = "https://official-joke-api.appspot.com/random_joke"

    try:
        response = requests.get(joke_url)
        response.raise_for_status()
        joke_data = response.json()
        joke = f"{joke_data['setup']} - {joke_data['punchline']}"
        logging.info(f"Random joke: {joke}")
        return joke
    except requests.exceptions.RequestException as e:
        logging.error(f"Error occurred while fetching joke: {str(e)}")
        return f"An error occurred while fetching a joke: {str(e)}"


# Define tools (functions) that can be called by the AI model
tools = [
    {
        "name": "get_news",
        "description": "Use this function to get the latest news from NewsApi. Only use this if the user has explicitly ask for news",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query phrase",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_current_weather",
        "description": "Use this function to search api.open-meteo.com to get weather from a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_random_joke",
        "description": "Get a random joke",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    DEFAULT_RESPONSE_FUNCTION,
]


####
## function calling tools v2
####
def get_top_headlines(query: str = None, country: str = None, category: str = None):
    """Retrieve top headlines from newsapi.org (API key required)"""

    base_url = "https://newsapi.org/v2/top-headlines"
    headers = {"x-api-key": newsapi_key}
    params = {"category": "general", "language": "en"}
    if query is not None:
        params["q"] = query
    if country is not None:
        params["country"] = country
    if category is not None:
        params["category"] = category

    # Fetch from newsapi.org - reference: https://newsapi.org/docs/endpoints/top-headlines
    response = requests.get(base_url, params=params, headers=headers)
    data = response.json()

    if data["status"] == "ok":
        print(f"Processing {data['totalResults']} articles from newsapi.org")
        return json.dumps(data["articles"])
    else:
        print("Request failed with message:", data["message"])
        return "No articles found"


def what_is_bigger(n, m):
    if n > m:
        return f"{n} is bigger"
    elif m > n:
        return f"{m} is bigger"
    else:
        return f"{n} and {m} are equal"


def get_current_time():
    """
    Get the current time in a more human-readable format.
    :return: The current time.
    """

    now = datetime.datetime.now()
    current_time = now.strftime("%I:%M:%S %p")  # Using 12-hour format with AM/PM
    current_date = now.strftime(
        "%A, %B %d, %Y"
    )  # Full weekday, month name, day, and year

    return f"Current Date and Time = {current_date}, {current_time}"


def get_current_weather_v2(city: str) -> str:
    """Get the current weather for a city
    Args:
        city: The city to get the weather for
    """
    base_url = f"http://wttr.in/{city}?format=j1"
    response = requests.get(base_url)
    data = response.json()
    return f"The current temperature in {city} is: {data['current_condition'][0]['temp_C']}°C"


def chat_with_ollama_no_functions(user_question):
    response = ollama.chat(
        model="dolphin-phi", messages=[{"role": "user", "content": user_question}]
    )
    return response


def vision_with_ollama(blobUpload, prompt):
    # Separate the metadata from the image data
    head, data = blobUpload.split(",", 1)
    # Get the file extension (gif, jpeg, png)
    file_ext = head.split(";")[0].split("/")[1]
    # Decode the image data
    plain_data = base64.b64decode(data)
    # Write the image to a file
    with open("image." + file_ext, "wb") as f:
        f.write(plain_data)
        print(f)
    # read image and encode to base64
    with open(f.name, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "moondream:v2",
        "prompt": prompt,
        "stream": False,
        "images": [image_base64],
    }
    # send that to llava
    response = requests.post(url, data=json.dumps(payload))
    print(response.json()["response"])
    content = response.json()["response"]
    return content


def chat_with_ollama(user_question):
    response = ollama.chat(
        model="qwen2",
        messages=[{"role": "user", "content": user_question}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_top_headlines",
                    "description": "Get top news headlines by country and/or category",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Freeform keywords or a phrase to search for.",
                            },
                            "country": {
                                "type": "string",
                                "description": "The 2-letter ISO 3166-1 code of the country you want to get headlines for",
                            },
                            "category": {
                                "type": "string",
                                "description": "The category you want to get headlines for",
                                "enum": [
                                    "business",
                                    "entertainment",
                                    "general",
                                    "health",
                                    "science",
                                    "sports",
                                    "technology",
                                ],
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather_v2",
                    "description": "Use this function to search api.open-meteo.com to get weather from a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time in a more human-readable format.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "which_is_bigger",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "float",
                            },
                            "m": {"type": "float"},
                        },
                        "required": ["n", "m"],
                    },
                },
            },
        ],
    )
    logging.info(f"response from chat with ollama: {response}")
    return response


####
## app pages
####


@app.route("/")
def index():
    return render_template("index.html", page="ChatGPT Project", **locals())


# function to extract html document from given url
def getHTMLdocument(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
    }
    # request for HTML document of given url
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        # response will be provided in JSON format
        return response.content
    except RequestException as e:
        msg = str(e)
        return msg


def image_handler(tag, specific_element, requested_url):
    image_paths = []
    if tag == "img":
        images = [img["src"] for img in specific_element]
        for i in specific_element:
            image_path = i.attrs["src"]
            if "base64" in image_path:
                print("base64 detected")
                image_path = i.attrs["data-src"]
            valid_imgpath = validators.url(image_path)
            if valid_imgpath == True:
                full_path = image_path
                image_paths.append(full_path)
            else:
                full_path = urljoin(requested_url, image_path)
                image_paths.append(full_path)

    return image_paths


@app.route("/scraper.html", methods=["POST", "GET"])
def scraper():
    if request.method == "POST":
        url = request.form["url"]
        # define a HTML document
        html = getHTMLdocument(request.form["url"])
        search_items = request.form.getlist("search_html_items")
        print(search_items)
        counter = len(search_items)
        # parse the HTML content with Beautiful Soup
        soup = BeautifulSoup(html, "html.parser")
        results = []
        if counter != 0:
            for i in search_items:
                if "img" in i:
                    specific_element = soup.find_all(i)
                    image_paths = image_handler(i, specific_element, url)
                if "video" in i:
                    specific_element = soup.find_all(i)
                    video_paths = image_handler(i, specific_element, url)
                if "all" in search_items:
                    # get all tags
                    tags = {tag.name for tag in soup.find_all()}
                    # iterate all tags
                    for tag in tags:
                        results.append(tags)
                else:
                    results.append(soup.find_all(i))
                    print("results", results)
        else:
            # print the HTML in a beautiful form
            scrapedpage = soup.prettify()

    return render_template("scraper.html", page="webscraping tool", **locals())


@app.route("/setup.html", methods=["POST", "GET"])
def setup():
    rows = show_system_instruction()
    model_lst = client.models.list()
    # print(model_lst)
    models = []
    for i in model_lst:
        model_id = i.id
        models.append(model_id)
    if request.method == "POST":
        session["chat_content"] = request.form.get("content_list")
        session["chat_model"] = request.form.get("model")

    return render_template("setup.html", page="Setup ChatGPT", **locals())


@app.route("/setup_v2.html", methods=["POST", "GET"])
def setup_v2():
    rows = show_system_instruction()
    model_lst = client.models.list()
    # print(model_lst)
    models = []
    for i in model_lst:
        if i.created:
            dt = datetime.datetime.fromtimestamp(i.created)
            i.created = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(i.created)
        # model_id=i.id
        # model_created=i.created
        # model_object=i.object
        # model_owned_by=i.owned_by
        models.append(i)
    if request.method == "POST":
        for i in request.values:
            print(i)
            if "gptcontent" in i:
                session["chat_content"] = request.form["gptcontent"]
            if "gptmodel" in i:
                session["chat_model"] = request.form["gptmodel"]
            else:
                session["chat_model"] = session["chat_model"]
                session["chat_content"] = session["chat_content"]
                print("nothing to change")

    return render_template("setup_v2.html", page="Setup ChatGPT", **locals())


@app.route("/setup_stabilityai.html", methods=["POST", "GET"])
def setup_stabilityai():
    getAccount = requests.get(
        "https://api.stability.ai/v1/user/account",
        headers={"Authorization": f"Bearer {api_key_stability}"},
    )
    getBalance = requests.get(
        "https://api.stability.ai/v1/user/balance",
        headers={"Authorization": f"Bearer {api_key_stability}"},
    )
    dataBalance = getBalance.json()
    dataAccount = getAccount.json()
    accountItems = []
    for i in dataAccount.items():
        accountItems.append(i)
    for i in dataBalance.items():
        accountItems.append(i)
    print(accountItems)
    engines = requests.get(
        "https://api.stability.ai/v1/engines/list",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key_stability}",
        },
    )
    model_lst = engines.json()
    models = []
    for i in model_lst:
        print(i["id"])
        model_id = i["id"]
        models.append(model_id)
    if request.method == "POST":
        session["chat_content_stability_ai"] = request.form.get(
            "content_list_stability_ai"
        )
        session["chat_model_stability_ai"] = request.form.get("model_stability_ai")

    return render_template("setup_stabilityai.html", page="Setup ChatGPT", **locals())


@app.route("/setup_stabilityaiv2.html", methods=["POST", "GET"])
def setup_stabilityaiv2():
    getAccount = requests.get(
        "https://api.stability.ai/v1/user/account",
        verify=False,
        headers={"Authorization": f"Bearer {api_key_stability}"},
    )
    getBalance = requests.get(
        "https://api.stability.ai/v1/user/balance",
        verify=False,
        headers={"Authorization": f"Bearer {api_key_stability}"},
    )
    getSystemStatus = requests.get(
        "https://stabilityai.instatus.com/summary.json",
        verify=False,
        headers={"Authorization": f"Bearer {api_key_stability}"},
    )
    systemStatus = getSystemStatus.json()
    print(type(systemStatus))
    dataBalance = getBalance.json()
    dataAccount = getAccount.json()
    accountItems = []
    systemStatusItems = []
    for i in dataAccount.items():
        accountItems.append(i)
    for i in dataBalance.items():
        accountItems.append(i)
    for i in systemStatus.items():
        systemStatusItems.append(i)
    print(systemStatusItems)
    engines = requests.get(
        "https://api.stability.ai/v1/engines/list",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key_stability}",
        },
    )
    model_lst = engines.json()
    models = []
    for i in model_lst:
        # print(i['id'])
        # model_id=i['id']
        models.append(i)
    if request.method == "POST":
        #  session["chat_content_stability_ai"] = request.form.get("content_list_stability_ai")
        session["chat_model_stability_ai"] = request.form["model_stability_ai"]

    return render_template("setup_stabilityaiv2.html", page="Setup ChatGPT", **locals())


@app.route("/instruction.html", methods=["POST", "GET"])
def instruction():
    rows = show_system_instruction()
    conn = get_db_connection()
    if "add_instruction" in request.form:
        content = request.form["content"]
        save_system_instruction(content)
        rows = show_system_instruction()
    if "update-instruction" in request.form:
        try:
            si_id = str.strip(request.form["instruction_id"])
            si_content = str.strip(request.form["instruction_content"])
            with conn as con:
                cur = con.cursor()
                cur.execute(
                    "UPDATE system_instruction SET content = ? WHERE id = ?",
                    [si_content, si_id],
                )
                con.commit()
                rows = show_system_instruction()
                msg = "Successfully Changed"
        except:
            conn.rollback()
            msg = "Unsuccessfull. Please retry later"
        finally:
            conn.close()
            return render_template(
                "instruction.html", page="list Instruction", **locals()
            )

    if "delete-instruction" in request.form:
        try:
            si_id = str.strip(request.form["instruction_id"])
            with conn as con:
                cur = con.cursor()
                cur.execute("DELETE from system_instruction WHERE id = ?", [si_id])
                con.commit()
                rows = show_system_instruction()
                msg = "Successfully Changed"
        except:
            conn.rollback()
            msg = "Unsuccessfull. Please retry later."
        finally:
            conn.close()
            return render_template(
                "instruction.html", page="list Instruction", **locals()
            )

    return render_template("instruction.html", page="list Instruction", **locals())


@app.route("/chatbot.html", methods=["POST", "GET"])
def chatbot():
    #    session_check=session["chat_content"]
    rows = show_system_instruction()
    if request.method == "POST":
        user_input = request.form["prompt"]
        moderation_check = moderate_text(user_input)
        print(moderation_check)
        messages = []
        # sets the role of chatgpt
        # system: facts, basic and important context (no behaviour instructions here): example: At this place apples are purple.
        if not session.get("chat_content"):
            messages.append(
                {
                    "role": "system",
                    "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-04-10",
                }
            )
        else:
            messages.append({"role": "system", "content": session.get("chat_content")})
        question = {}
        question["role"] = "user"
        question["content"] = user_input
        messages.append(question)
        print(messages)
        if not session.get("chat_model"):
            model = "gpt-3.5-turbo"
        else:
            model = session["chat_model"]
        try:
            response = client.chat.completions.create(
                # model="gpt-4",
                model=model,
                messages=messages,
                temperature=0.5,
            )
            content = response.choices[0].message.content.replace("\n", "<br>")
            # content = response.choices[0].message["content"]
            # content = response['choices'][0]['message']['content'].replace('\n', '<br>')
            count_tokens = response.usage.total_tokens
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            content = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"
        finally:
            save_response_token(count_tokens)

        return jsonify(content=content, count_tokens=count_tokens), 200

    return render_template("chatbot.html", page="Chatbot project", **locals())


@app.route("/chatbot_v2.html", methods=["POST", "GET"])
def chatbot_v2():
    model_lst = client.models.list()
    # print(model_lst)
    models = []
    for i in model_lst:
        model_id = i.id
        models.append(model_id)
    if request.method == "POST":
        user_input = request.form["prompt"]
        moderation_check = moderate_text(user_input)
        print(moderation_check)
        messages = []
        # sets the role of chatgpt
        # system: facts, basic and important context (no behaviour instructions here): example: At this place apples are purple.
        if not session.get("chat_content"):
            messages.append(
                {
                    "role": "system",
                    "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-04-10",
                }
            )
        else:
            messages.append({"role": "system", "content": session.get("chat_content")})
        question = {}
        question["role"] = "user"
        question["content"] = user_input
        messages.append(question)
        print(messages)
        if not session.get("chat_model"):
            model = "gpt-3.5-turbo"
        else:
            model = session["chat_model"]
        try:
            response = client.chat.completions.create(
                # model="gpt-4",
                model=model,
                messages=messages,
                temperature=0.5,
            )
            content = response.choices[0].message.content.replace("\n", "<br>")
            # content = response.choices[0].message["content"]
            # content = response['choices'][0]['message']['content'].replace('\n', '<br>')
            count_tokens = response.usage.total_tokens
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"
        except openai.Error as e:
            # Handle OpenAI API errors
            error_message = str(e)
            return jsonify({"error": error_message}), 500
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            content = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"
        finally:
            save_response_token(count_tokens)

        return jsonify(content=content, count_tokens=count_tokens), 200
    return render_template("chatbot_v2.html", page="Chatbot project v2", **locals())


@app.route("/chatbot_ollama.html", methods=["POST", "GET"])
def chatbot_ollama():
    local_model_list = listInstalledModels()
    model_list = []
    for key, value in local_model_list.items():
        for i in value:
            model_list.append(i)
    if request.method == "POST":
        user_input = request.form["prompt"]
        gptmodel = request.form["gptmodel"]
        try:
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            )

            response = client.chat.completions.create(
                model=gptmodel,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
            )
            content = response.choices[0].message.content.replace("\n", "<br>")
            # content = response.choices[0].message["content"]
            # content = response['choices'][0]['message']['content'].replace('\n', '<br>')
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}."
            return jsonify(content=content), 200
        except openai.Error as e:
            # Handle OpenAI API errors
            content = str(e)
            return jsonify(content=content), 200
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            content = f"OpenAI API returned an API Error: {e}"
            return jsonify(content=content), 200
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            content = f"OpenAI API request exceeded rate limit: {e}. "
            return jsonify(content=content), 200
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            content = f"OpenAI API request was not authorized: {e}"
            return jsonify(content=content), 200
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            content = f"OpenAI API request timed out: {e}"
            return jsonify(content=content), 200
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            content = f"OpenAI API request failed to connect: {e}"
            return jsonify(content=content), 200
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            content = f"OpenAI API request was invalid: {e}"
            return jsonify(content=content), 200
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"
            return jsonify(content=content), 200
        except openai.error.BadRequestError:
            # Handle bad request error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"
            return jsonify(content=content), 200

        return jsonify(content=content), 200
    return render_template("chatbot_ollama.html", page="Chatbot Ollama", **locals())


@app.route("/ollama_chat_history.html", methods=["POST", "GET"])
def ollama_chat_history():
    local_model_list = listInstalledModels()
    model_list = []
    for key, value in local_model_list.items():
        for i in value:
            model_list.append(i)
    if request.method == "POST":
        user_input = request.form["prompt"]
        print(user_input)
        try:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant.",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )
            chain = prompt_template | llm
            response = chain.invoke({"input": user_input, "chat_history": chat_history})
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            contentToAjax = response.replace("\n", "<br>")
            print(contentToAjax)
            # content = response.choices[0].message["content"]
            # content = response['choices'][0]['message']['content'].replace('\n', '<br>')
        except Exception as e:
            print(e)
            contentToAjax = f"Oops something went wrong.: {sys.exc_info()[0]}"
            return jsonify({"error": contentToAjax}), 500
        except openai.Error as e:
            # Handle OpenAI API errors
            error_message = str(e)
            return jsonify({"error": error_message}), 500
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            contentToAjax = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            contentToAjax = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            contentToAjax = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            contentToAjax = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            contentToAjax = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            contentToAjax = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            contentToAjax = f"OpenAI API request was not permitted: {e}"

        return jsonify(content=contentToAjax), 200
    return render_template(
        "ollama_chat_history.html", page="Ollama Chatbot History", **locals()
    )


@app.route("/ollama_function_call.html", methods=["POST", "GET"])
def ollama_function_call():

    local_model_list = listInstalledModels()
    model_list = []
    for key, value in local_model_list.items():
        for i in value:
            model_list.append(i)
    if request.method == "POST":
        # Initialize the AI model
        model_fc = OllamaFunctions(model="dolphin-phi", format="json", temperature=0)
        model_fc = model_fc.bind_tools(tools=tools)
        user_input = request.form["prompt"]
        print(user_input)
        try:
            # Create the prompt template for the AI model
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are a helpful AI assistant. Use the provided tools when necessary. If the request does not fit any tools try to answer the request as you normally do."
                    ),
                    ("human", "{input}"),
                ]
            )
            formatted_prompt = prompt_template.format_messages(input=user_input)
            result = model_fc.invoke(formatted_prompt)
            print(" look", result.tool_calls)
            if result.tool_calls:
                for tool_call in result.tool_calls:
                    function_name = tool_call["name"]
                    args = tool_call["args"]
                    if function_name == "get_current_weather":
                        result = get_current_weather(**args)
                    if function_name == "get_news":
                        result = get_news(user_input)
                    if function_name == "get_random_joke":
                        result = get_random_joke()
            if isinstance(result, list):
                contentToAjax = result
            else:
                contentToAjax = result.replace("\n", "<br>")
            print(contentToAjax)
            # content = response.choices[0].message["content"]
            # content = response['choices'][0]['message']['content'].replace('\n', '<br>')
        except Exception as e:
            print(e)
            contentToAjax = f"Oops something went wrong.: {sys.exc_info()[0]}"
            return jsonify({"error": contentToAjax}), 500
        except openai.Error as e:
            # Handle OpenAI API errors
            error_message = str(e)
            return jsonify({"error": error_message}), 500
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            contentToAjax = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            contentToAjax = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            contentToAjax = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            contentToAjax = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            contentToAjax = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            contentToAjax = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            contentToAjax = f"OpenAI API request was not permitted: {e}"

        return jsonify(content=contentToAjax), 200
    return render_template(
        "ollama_function_call.html", page="Ollama Function Call", **locals()
    )


@app.route("/ollama_function_call_v2.html", methods=["POST", "GET"])
def ollama_function_call_v2():
    local_model_list = listInstalledModels()
    model_list = []
    for key, value in local_model_list.items():
        for i in value:
            model_list.append(i)
    if request.method == "POST":
        # question to AI
        user_input = request.form["prompt"]
        try:
            if "image" in request.form:
                # image data
                blobUpload = request.form["image"]
                response = vision_with_ollama(blobUpload, user_input)
                contentToAjax = response
                print("image response")
                # # Separate the metadata from the image data
                # head, data = blobUpload.split(",", 1)
                # # Get the file extension (gif, jpeg, png)
                # file_ext = head.split(";")[0].split("/")[1]
                # # Decode the image data
                # plain_data = base64.b64decode(data)
                # # Write the image to a file
                # with open("image." + file_ext, "wb") as f:
                #     f.write(plain_data)
                #     print(f.name)
            else:
                response = chat_with_ollama(user_input)
                logging.info(f"response from function call: {response}")
                if (
                    "tool_calls" in response["message"]
                    and response["message"]["tool_calls"]
                ):
                    tools_calls = response["message"]["tool_calls"]
                    logging.info(f"tools_calls var: {tools_calls}")
                    for tool_call in tools_calls:
                        tool_name = tool_call["function"]["name"]
                        arguments = tool_call["function"]["arguments"]
                        logging.info(f"arguments: {arguments}")
                        if (
                            tool_name == "get_current_weather_v2"
                            and "city" in arguments
                            and arguments is not None
                        ):
                            result = get_current_weather_v2(arguments["city"])
                            print("Weather function result:", result)
                            contentToAjax = result
                        elif tool_name == "get_current_time":
                            result = get_current_time()
                            print("Current time function result:", result)
                            contentToAjax = result
                        elif (
                            tool_name == "get_top_headlines"
                            and "query" in arguments
                            and "country" in arguments
                            and "category" in arguments
                            and arguments is not None
                        ):
                            result = get_top_headlines(
                                arguments["query"],
                                arguments["country"],
                                arguments["category"],
                            )
                            print("News function result:", result)
                            contentToAjax = result
                        elif (
                            tool_name == "which_is_bigger"
                            and "n" in arguments
                            and "m" in arguments
                        ):
                            n, m = float(arguments["n"]), float(arguments["m"])
                            result = what_is_bigger(n, m)
                            print("Comparison function result:", result)
                            contentToAjax = result
                        else:
                            logging.info(
                                f"No valid arguments found for function: {tool_name}"
                            )
                            # If no tool calls or no valid arguments, use the LLM's response
                            response = chat_with_ollama_no_functions(user_input)
                            logging.info(f"no function calls: {response}")
                            print("AI response:", response["message"]["content"])
                            contentToAjax = response["message"]["content"].replace(
                                "\n", "<br>"
                            )
                else:
                    print("chat_with_ollama_no_functions")
                    # If no tool calls or no valid arguments, use the LLM's response
                    response = chat_with_ollama_no_functions(user_input)
                    logging.info(f"no function calls: {response}")
                    print("AI response:", response["message"]["content"])
                    contentToAjax = response["message"]["content"].replace("\n", "<br>")
        except Exception as e:
            print(e)
            contentToAjax = f"Oops something went wrong.: {sys.exc_info()[0]}"
            return jsonify({"error": contentToAjax}), 500
        except openai.Error as e:
            # Handle OpenAI API errors
            error_message = str(e)
            return jsonify({"error": error_message}), 500
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            contentToAjax = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            contentToAjax = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            contentToAjax = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            contentToAjax = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            contentToAjax = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            contentToAjax = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            contentToAjax = f"OpenAI API request was not permitted: {e}"

        return jsonify(content=contentToAjax), 200
    return render_template(
        "ollama_function_call_v2.html", page="Ollama Function Call v2", **locals()
    )


@app.route("/embedding_ollama.html", methods=["POST", "GET"])
def embedding_ollama():
    local_model_list = listInstalledModels()
    model_list = []
    for key, value in local_model_list.items():
        for i in value:
            model_list.append(i)
    if request.method == "POST":
        try:
            # 0. Get data
            urls = request.form.get("user_input_url").split(",")
            user_input = request.form.get("prompt")
            user_input_file = request.form.getlist("user_input_file[]")
            if len(user_input_file) >= 1:
                for file in user_input_file:
                    if allowed_file(file) is True:
                        print("true")
                        # 1. Split data into chunks for PDF
                        loader = PyPDFLoader(
                            f"/home/alexaj14/Documents/AI-Project/static/pdf/{file}"
                        )
                        doc_splits = loader.load_and_split()
                    else:
                        print("no need to go further")
                        content = "Only PDF are accepted"
                        return jsonify(content=content), 404
            else:
                # 1. Split data into chunks url
                docs = [WebBaseLoader(url, verify_ssl=False).load() for url in urls]
                docs_list = [item for sublist in docs for item in sublist]
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=7500, chunk_overlap=100
                )
                doc_splits = text_splitter.split_documents(docs_list)

            # 2. Convert documents to Embeddings and store them
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=embedding,
            )
            retriever = vectorstore.as_retriever()

            # 3. Retrieval-Augmented Generation
            after_rag_template = """You are an respectful and honest assistant. You have to answer the user's \
            questions using only the context provided to you. If you don't know the answer, \
            just say you don't know. Don't try to make up an answer.:
            {context}
            Question: {question}
            """
            after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
            after_rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | after_rag_prompt
                | model_local
                | StrOutputParser()
            )
            content = after_rag_chain.invoke(user_input)
        except Exception as e:
            print(e)
            return jsonify(content=e), 404
        return jsonify(content=content), 200

    return render_template("embedding_ollama.html", page="Embedding Ollama", **locals())


"""
New databases start with a collection name of 'privategpt' if the user does not choose something else.
All existing Chroma databases will be converged to 'privategpt'. 
They have an initial collection when the db is created. 
Because we get Chroma from langchain, they set a default of 'langchain'. I would migrate the existing databases with a collection entry of 'langchain' to rename it to 'privategpt' when initially discovered, making 'privategpt' the default if the user does not choose otherwise.
The users preferred collection setting should be persistent until explicitly changed by the user. 
This implies persistent user prefs (which might be a bigger issue). 
In the absence of prefs and if one collection and only one collection exists, the choice is obvious. 
A command line setting should be supported in the interim.
When the user indicates (UI dependent), he should be offered a selection of all current collections and the opportunity to define a new one. 
The user should be able to change the current collection at any time. This appears to be low impact, but I don't know if it is effectively zero impact.
Additional facilities should allow the deletion of a collection (loss of ingest effort) or the renaming of a collection. Note that there are restrictions on the construction of valid names (from the Chroma docs):
Chroma uses collection names in the url, so there are a few restrictions on naming them:
The length of the name must be between 3 and 63 characters.
The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.
The name must not contain two consecutive dots.
The name must not be a valid IP address.                                                                                                                                                                                                                                                                                                                                                                                                         ||
"""


@app.route("/chroma_ingest.html", methods=["POST", "GET"])
def chroma_ingest():
    if request.method == "POST":
        try:
            # 0. Get data
            urls = request.form.get("user_input_url").split(",")
            collection_name_from_form = request.form.get("collection_name")
            user_input = request.form.get("prompt")
            user_input_file = request.form.getlist("user_input_file[]")
            if len(user_input_file) >= 1:
                for file in user_input_file:
                    if allowed_file(file) is True:
                        chunk_size = 1024
                        chunk_overlap = 80
                        print("true pdf")
                        # 1. Split data into chunks for PDF
                        loader = PyPDFLoader(
                            f"/home/alexaj14/Documents/AI-Project/static/pdf/{file}"
                        )
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            is_separator_regex=False,
                        )
                        doc_splits = loader.load_and_split(text_splitter)
                    else:
                        print("no need to go further")
                        content = "Only PDF are accepted"
                        return jsonify(content=content), 200
            else:
                # 1. Split data into chunks url
                chunk_size = 1024
                chunk_overlap = 80
                docs = [WebBaseLoader(url, verify_ssl=False).load() for url in urls]
                docs_list = [item for sublist in docs for item in sublist]
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
                doc_splits = text_splitter.split_documents(docs_list)

            # 2. Convert documents to Embeddings and store them
            # collection_metadata={"hnsw:M": 1024,"hnsw:ef": 64}
            # collection_metadata={"hnsw:space": "cosine"}
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=collection_name_from_form,
                persist_directory=f"./vectorDatabase/{collection_name_from_form}",
                embedding=embedding,
            )
            # Save the embedding to disk
            # vectorstore.persist()
            global Chromaclient_specific
            get_chroma_client_specific(collection_name_from_form)
            collections_specific = Chromaclient_specific.list_collections()
            collections_from_chroma_compare = []
            for coll in collections_specific:
                # print(coll.name)
                collections_from_chroma_compare.append(str(coll.id))
                if collection_name_from_form in coll.name:
                    collection_name_to_chroma = coll.name
                    uuid = coll.id
                    metadata = str(coll.metadata)
                    print(metadata)
            if len(collection_name_to_chroma) > 0:
                content = "Documents saved to ChromaCollection"
            else:
                content = "Problem with saving data to ChromaCollection"

        except Exception as e:
            print("Exception", e)
            e = str(e)
            return jsonify(content=e), 200
        else:
            semantic = "No"
            store_chroma_in_sql_collection(
                collection_name_from_form,
                str(uuid),
                metadata,
                int(chunk_size),
                int(chunk_overlap),
                embed_model,
                str(collection_name_from_form),
                str(urls),
                str(user_input_file),
                semantic,
            )

        return jsonify(content=content), 200
    return render_template("chroma_ingest.html", page="chroma_ingest", **locals())


@app.route("/chunking_strategies.html", methods=["POST", "GET"])
def chunking_strategies():
    if request.method == "POST":
        # 0. Get data
        chunking_strategy_from_form = request.form["chunking_strategy"]
        collection_name_from_form = request.form["collection_name"]
        user_input = request.form["prompt"]
        try:
            logging.info(f"Chunking Staretgy used: {chunking_strategy_from_form}")
            if "fixed_size_chunking" in chunking_strategy_from_form:
                chunk_size = 256
                chunk_overlap = 20
                text_splitter = CharacterTextSplitter(
                    separator="\n\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                doc_splits = text_splitter.create_documents([user_input])

            elif "naive" in chunking_strategy_from_form:
                chunk_size = 0
                chunk_overlap = 0
                text_splitter = CharacterTextSplitter(separator="\n\n")
                doc_splits = text_splitter.create_documents([user_input])

            elif "NLTK" in chunking_strategy_from_form:
                chunk_size = 0
                chunk_overlap = 0
                text_splitter = NLTKTextSplitter()
                doc_splits = text_splitter.create_documents([user_input])

            elif "spacy" in chunking_strategy_from_form:
                chunk_size = 0
                chunk_overlap = 0
                text_splitter = SpacyTextSplitter()
                doc_splits = text_splitter.create_documents([user_input])

            elif "python" in chunking_strategy_from_form:
                chunk_size = 30
                chunk_overlap = 0
                text_splitter = PythonCodeTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                doc_splits = text_splitter.create_documents([user_input])

            else:
                print("no other strategy devoloped.")
                content = "No strategy found. stop here"
                return jsonify(content=content), 200

            # 2. Convert documents to Embeddings and store them
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=collection_name_from_form,
                persist_directory=f"./vectorDatabase/{collection_name_from_form}",
                embedding=embedding,
            )
            # Save the embedding to disk
            # vectorstore.persist()
            global Chromaclient_specific
            get_chroma_client_specific(collection_name_from_form)
            collections_specific = Chromaclient_specific.list_collections()
            collections_from_chroma_compare = []
            for coll in collections_specific:
                # print(coll.name)
                collections_from_chroma_compare.append(str(coll.id))
                if collection_name_from_form in coll.name:
                    collection_name_to_chroma = coll.name
                    uuid = coll.id
                    metadata = str(coll.metadata)
                    print(metadata)
            if len(collection_name_to_chroma) > 0:
                content = "Documents saved to ChromaCollection"
            else:
                content = "Problem with saving data to ChromaCollection"
        except Exception as e:
            print("Exception", e)
            e = str(e)
            return jsonify(content=e), 200
        else:
            semantic = chunking_strategy_from_form
            urls = "None"
            user_input_file = "None"
            store_chroma_in_sql_collection(
                collection_name_from_form,
                str(uuid),
                metadata,
                int(chunk_size),
                int(chunk_overlap),
                embed_model,
                str(collection_name_from_form),
                str(urls),
                str(user_input_file),
                semantic,
            )

        return jsonify(content=content), 200
    return render_template(
        "chunking_strategies.html", page="Chunking Strategies", **locals()
    )


@app.route("/chroma_view_database.html", methods=["POST", "GET"])
def chroma_view_database():
    collections_from_sql = show_sql_collections()
    collections_from_sql_names = []
    for i in collections_from_sql:
        collections_from_sql_names.append(i["collection_name"])
    if request.method == "POST":
        if "selected_collection" in request.form:
            try:
                collection_name_from_form = request.form.get(
                    "collection_name_from_form"
                )
                global Chromaclient_specific
                get_chroma_client_specific(collection_name_from_form)
                selected_collection = Chromaclient_specific.get_collection(
                    name=collection_name_from_form
                )
                content = selected_collection.get()
                return jsonify(content=content), 200
            except Exception as error:
                content = "Internal Sever Error {}".format(error)
                return jsonify(content=content), 404
        if "delete_collection" in request.form:
            get_selected_collection = request.form.get("get_selected_collection")
            content = sync_local_with_chroma_db(get_selected_collection)
            return jsonify(content=content), 200

    return render_template(
        "chroma_view_database.html", page="chroma_view_database", **locals()
    )


@app.route("/load_embedding_ollama.html", methods=["POST", "GET"])
def load_embedding_ollama():
    collections_from_sql = show_sql_collections()
    local_model_list = listInstalledModels()
    model_list = []
    for key, value in local_model_list.items():
        for i in value:
            model_list.append(i)
    if request.method == "POST":
        selected_model = request.form["selected_model"]
        print("selected model", selected_model)
        if "default" in selected_model:
            model_local = ChatOllama(model="dolphin-phi")
        else:
            model_local = ChatOllama(model=selected_model)
        system_prompt = request.form["system_prompt"]
        if "save_system_prompt" in request.form:
            print("save_system_instruction")
            try:
                save_system_instruction(system_prompt)
            except Exception as e:
                print(e)
                return jsonify(content=e), 404
        if "ask_rag" in request.form:
            print("ask_rag")
            get_selected_collection = request.form["get_selected_collection"]

            try:
                # 0. Get data
                prompt = request.form["prompt"]
                # 1.retreive data from chromadb
                get_collection_from_chroma = Chroma(
                    collection_name=get_selected_collection,
                    persist_directory=f"./vectorDatabase/{get_selected_collection}",
                    embedding_function=embedding,
                )
                retriever = get_collection_from_chroma.as_retriever()
                # 2. Retrieval-Augmented Generation
                if len(system_prompt) > 0:
                    print("use system prompt")
                    after_rag_template = """{system_prompt}
                    {context}
                    Question: {question}
                    """

                    def get_system_prompt(_):
                        return system_prompt

                    after_rag_prompt = ChatPromptTemplate.from_template(
                        after_rag_template
                    )
                    after_rag_chain = (
                        {
                            "context": retriever,
                            "question": RunnablePassthrough(),
                            "system_prompt": RunnableLambda(get_system_prompt),
                        }
                        | after_rag_prompt
                        | model_local
                        | StrOutputParser()
                    )
                else:
                    print("default prompt")
                    after_rag_template = """You are an respectful and honest assistant. You have to answer the user's \
                    questions using only the context provided to you. If you don't know the answer, \
                    just say you don't know. Don't try to make up an answer.:
                    {context}
                    Question: {question}
                    """
                    after_rag_prompt = ChatPromptTemplate.from_template(
                        after_rag_template
                    )
                    after_rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | after_rag_prompt
                        | model_local
                        | StrOutputParser()
                    )

                content = after_rag_chain.invoke(prompt).replace("\n", "<br>")
                # print("this is the content", content)
            except Exception as e:
                print(e)
                return jsonify(content=e), 404
            finally:
                return jsonify(content=content), 200
        else:
            content = "Error. Please check your input"
            return jsonify(content=content), 404

    return render_template(
        "load_embedding_ollama.html", page="Load-Embedding", **locals()
    )


def sync_local_with_chroma_db(i):
    try:
        collection = Chromaclient_specific.get_collection(name=i)
        # Get all documents in the collection
        db_data = collection.get()
        # Extract metadata
        metadatas = db_data["metadatas"]
        ids = db_data["ids"]
        # Display all source file names present in the collection
        print("Source file names present inside the collection:")
        source_file_names = set(metadata.get("source") for metadata in metadatas)
        print("source_file_names", source_file_names)
        if len(source_file_names) < 1:
            # Deleting the collection from chromadb
            Chromaclient_specific.delete_collection(name=collection.name)
            print(
                f"Collection'{collection.name}' without source file has been deleted."
            )
            content = (
                f"Collection'{collection.name}' without source file has been deleted."
            )
        else:
            for source_file_name in source_file_names:
                id_to_delete = [
                    id
                    for id, metadata in zip(ids, metadatas)
                    if metadata.get("source") == source_file_name
                ]
                # Delete the documents with matching IDs from chromadb
                collection.delete(ids=id_to_delete)
                print("Documents have been deleted from the collection.")
                # Deleting the collection from chromadb
                Chromaclient_specific.delete_collection(name=collection.name)
                print(f"Collection '{collection.name}' has been deleted.")
                content = f"Collection '{collection.name}' has been deleted."
    except Exception as e:
        content = str(e)
        print("error is here", e)
    else:
        delete_not_stored_chromadb_collection(i)
        shutil.rmtree(f"./vectorDatabase/{i}", ignore_errors=True)
    return content


# @app.route('/voice', methods=['POST', 'GET'])
# def voice():
#       text_input = request.form['text_input']
#       # os.system('espeak -s 90 -p 50 -v en-us+f3 "'+text_input+'"')
#       # Set properties _before_ you add things to say
#       engine.setProperty('rate', 70)    # Speed percent (can go over 100)
#       engine.setProperty('volume', 0.9)  # Volume 0-1
#       # set the desired voice
#       voices = engine.getProperty('voices')
#       for voice in voices:
#         if voice.languages[0] == u'en_US':
#             engine.setProperty('voice', voice.id)
#             break

#       engine.say(text_input)
#       engine.runAndWait()

#       return render_template('chatbot.html', page="Chatbot project",**locals())


@app.route("/img_variation/<path:filename>", methods=["GET", "POST"])
def download(filename):
    uploads = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])
    return send_from_directory(directory=uploads, filename=filename, as_attachment=True)


@app.route("/imagevariation.html", methods=["POST", "GET"])
def imagevariation():
    image_names = os.listdir(IMG_FOLDER)
    print(image_names)
    image_path = []
    for image in image_names:
        abs_file_path = os.path.join(IMG_FOLDER, image)
        image_path.append(abs_file_path)
    # print(image_path)
    if request.method == "POST":
        user_input = request.form["prompt"]
        print(user_input)
        # Read the image file from disk and resize it
        image = Image.open(user_input)
        width, height = 256, 256
        image = image.resize((width, height))

        # Convert the image to a BytesIO object
        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        byte_array = byte_stream.getvalue()
        try:
            # response = openai.Image.create_variation(image=open(user_input, "rb"),n=1,size="1024x1024")
            response = client.images.create_variation(
                model="dall-e-2", image=byte_array, n=1, size="512x512"
            )
            content = response.data[0].url
            revised_prompt = response.data[0].revised_prompt
            print(content)
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            content = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"

        return jsonify(content=content), 200

    return render_template("imagevariation.html", page="Image Variation", **locals())


@app.route("/imagevariation_v2.html", methods=["POST", "GET"])
def imagevariation_v2():
    image_names = os.listdir(IMG_FOLDER)
    print(image_names)
    image_path = []
    for image in image_names:
        abs_file_path = os.path.join(IMG_FOLDER, image)
        image_path.append(abs_file_path)
    # print(image_path)
    if request.method == "POST":
        user_input = request.form["prompt"]
        print(user_input)
        # Read the image file from disk and resize it
        image = Image.open(user_input)
        width, height = 256, 256
        image = image.resize((width, height))

        # Convert the image to a BytesIO object
        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        byte_array = byte_stream.getvalue()
        try:
            # response = openai.Image.create_variation(image=open(user_input, "rb"),n=1,size="1024x1024")
            response = client.images.create_variation(
                model="dall-e-2", image=byte_array, n=1, size="512x512"
            )
            content = response.data[0].url
            revised_prompt = response.data[0].revised_prompt
            print(content)
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            content = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"

        return jsonify(content=content), 200

    return render_template("imagevariation_v2.html", page="Image Variation", **locals())


@app.route("/vision_ollama_v2.html", methods=["POST", "GET"])
def vision_ollama_v2():
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            # image data
            blobUpload = request.form["image"]
            response = vision_with_ollama(blobUpload, prompt)
            # contentToAjax = response
            print("image response")
            content = response
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"

        return jsonify(content=content), 200

    return render_template("vision_ollama_v2.html", page="Vision Model", **locals())


@app.route("/vision_ollama.html", methods=["POST", "GET"])
def vision_ollama():
    image_names = os.listdir(IMG_FOLDER)
    image_path = []
    for image in image_names:
        abs_file_path = os.path.join(IMG_FOLDER, image)
        image_path.append(abs_file_path)
    # print(image_path)
    if request.method == "POST":
        prompt = request.form["prompt"]
        # image file path
        selected_image_path = request.form["selected_image_path"]
        try:
            # read image and encode to base64
            with open(selected_image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llava:7b",
                "prompt": prompt,
                "stream": False,
                "images": [image_base64],
            }
            # send that to llava
            response = requests.post(url, data=json.dumps(payload))
            print(response.json()["response"])
            content = response.json()["response"]
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"

        return jsonify(content=content), 200

    return render_template("vision_ollama.html", page="Vision Model", **locals())


@app.route("/video_gallery.html", methods=["POST", "GET"])
def video_gallery():
    video_detail = show_generation_id()
    video_names = os.listdir(VIDEO_FOLDER)
    print(video_names)
    video_path = []
    for video in video_names:
        abs_file_path = os.path.join(VIDEO_FOLDER, video)
        video_path.append(abs_file_path)

    return render_template("video_gallery.html", page="Video Gallery", **locals())


@app.route("/img_generation.html", methods=["POST", "GET"])
def img_generation():
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_stability}",
                    "accept": "image/*",
                },
                files={"none": ""},
                data={
                    "prompt": prompt,
                    "output_format": "png",
                },
            )
            if response.status_code == 200:
                with open("static/img_generation/img_generation.png", "wb") as file:
                    file.write(response.content)
                    content = "img_generation.png"
                    return jsonify(content=content), 200
            else:
                raise Exception(str(response.json()))
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {response.text}"
            # raise Exception("Non-200 response: " + str(response.text))

        return jsonify(content=content), 200
    return render_template("img_generation.html", page="Image generation", **locals())


@app.route("/imgvar_stabilityai.html", methods=["POST", "GET"])
def imgvar_stabilityai():
    image_names = os.listdir(IMG_FOLDER)
    print(image_names)
    image_path = []
    for image in image_names:
        abs_file_path = os.path.join(IMG_FOLDER, image)
        image = Image.open(abs_file_path)
        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
        image.save(abs_file_path, "JPEG")
        image_path.append(abs_file_path)
    # print(image_path)
    if request.method == "POST":
        user_input = request.form["prompt"]
        if not session.get("chat_model_stability_ai"):
            engine = "stable-diffusion-v1-6"
        else:
            engine = session["chat_model_stability_ai"]
        print(user_input)
        try:
            response = requests.post(
                f"https://api.stability.ai/v1/generation/{engine}/image-to-image",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_stability}",
                },
                files={"init_image": open(user_input, "rb")},
                data={
                    "init_image_mode": "IMAGE_STRENGTH",
                    # How much influence the init_image has on the diffusion process.
                    # Values close to 1 will yield images very similar to the init_image while values
                    # close to 0 will yield images wildly different than the init_image.
                    "image_strength": 0.45,
                    # Number of diffusion steps to run
                    "steps": 40,
                    # Random noise seed (omit this option or use 0 for a random seed)
                    "seed": 0,
                    # How strictly the diffusion process adheres to the prompt text
                    # (higher values keep your image closer to your prompt)
                    "cfg_scale": 8,
                    # Pass in a style preset to guide the image model towards a particular style.
                    # This list of style presets is subject to change.
                    # see list below
                    # "style_preset": 'photographic',
                    # Number of images to generate
                    "samples": 1,
                    # An array of text prompts to use for generation.
                    "text_prompts[0][text]": "Create an image variation that enhances the given image",
                    "text_prompts[0][weight]": 1,
                    # negative prompt stating aspect you do not want in the result.
                    # below can be read as not blurry, notbad
                    "text_prompts[1][text]": "blurry, bad",
                    "text_prompts[1][weight]": -1,
                },
            )
            data = response.json()
            for i, image in enumerate(data["artifacts"]):
                with open(f'./static/out/img2img_{image["seed"]}.png', "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
                content = f'img2img_{image["seed"]}.png'
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {response.text}"
            # raise Exception("Non-200 response: " + str(response.text))

        return jsonify(content=content), 200

    return render_template(
        "imgvar_stabilityai.html", page="Image Variation", **locals()
    )


@app.route("/imgvar_stabilityai_v2.html", methods=["POST", "GET"])
def imgvar_stabilityai_v2():
    image_names = os.listdir(IMG_FOLDER)
    image_path = []
    for image in image_names:
        abs_file_path = os.path.join(IMG_FOLDER, image)
        image = Image.open(abs_file_path)
        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
        image.save(abs_file_path, "JPEG")
        image_path.append(abs_file_path)
    if request.method == "POST":
        user_input = request.form["prompt"]
        style_preset = request.form["style_preset"]
        image_strength = request.form["image_strength"]
        steps = request.form["steps"]
        if not session.get("chat_model_stability_ai"):
            engine = "stable-diffusion-v1-6"
        else:
            engine = session["chat_model_stability_ai"]
        #  print(user_input)
        prepare_data = {
            "init_image_mode": "IMAGE_STRENGTH",
            # How much influence the init_image has on the diffusion process.
            # Values close to 1 will yield images very similar to the init_image while values
            # close to 0 will yield images wildly different than the init_image.
            "image_strength": float(image_strength),
            # Number of diffusion steps to run
            "steps": int(steps),
            # Random noise seed (omit this option or use 0 for a random seed)
            "seed": 0,
            # How strictly the diffusion process adheres to the prompt text
            # (higher values keep your image closer to your prompt)
            "cfg_scale": 8,
            # Pass in a style preset to guide the image model towards a particular style.
            # This list of style presets is subject to change.
            # see list below
            # "style_preset":style_preset,
            # Number of images to generate
            "samples": 1,
            # An array of text prompts to use for generation.
            "text_prompts[0][text]": "Create an image variation that enhances the given image",
            "text_prompts[0][weight]": 1,
            # negative prompt stating aspect you do not want in the result.
            # below can be read as not blurry, notbad
            "text_prompts[1][text]": "blurry, bad",
            "text_prompts[1][weight]": -1,
        }
        if len(style_preset) > 0:
            prepare_data["style_preset"] = style_preset
            print(prepare_data)
        else:
            prepare_data = prepare_data
            print(prepare_data)
        try:
            response = requests.post(
                f"https://api.stability.ai/v1/generation/{engine}/image-to-image",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key_stability}",
                },
                files={"init_image": open(user_input, "rb")},
                data=prepare_data,
            )
            data = response.json()
            for i, image in enumerate(data["artifacts"]):
                with open(f'./static/out/img2img_{image["seed"]}.png', "wb") as f:
                    f.write(base64.b64decode(image["base64"]))
                content = f'img2img_{image["seed"]}.png'
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {response.text}"
            # raise Exception("Non-200 response: " + str(response.text))

        return jsonify(content=content), 200

    return render_template(
        "imgvar_stabilityai_v2.html", page="Image Variation", **locals()
    )


###stability AI###
# style preset values
# # Enum: # #
# None
# 3d-model
# analog-film
# anime
# cinematic
# comic-book
# digital-art
# enhance
# fantasy-art
# isometric
# line-art
# low-poly
# modeling-compound
# neon-punk
# origami
# photographic
# pixel-art
# tile-texture


@app.route("/imageToVideo_stabilityai.html", methods=["POST", "GET"])
def imageToVideo_stabilityai():
    is_retreived = show_generation_id()
    image_names = os.listdir(IMG_FOLDER)
    image_path = []
    for image in image_names:
        abs_file_path = os.path.join(IMG_FOLDER, image)
        image = Image.open(abs_file_path)
        image = image.resize((int(640), int(480)), resample=Image.LANCZOS)
        image.save(abs_file_path, "JPEG")
        image_path.append(abs_file_path)

    if request.method == "POST":
        print(request.form)
        if "prompt" in request.form:
            user_input = request.form["prompt"]
            seed = request.form["seed"]
            cfg_scale = request.form["cfg_scale"]
            motion_bucket_id = request.form["motion_bucket_id"]
            prepare_data = {
                # A specific value that is used to guide the 'randomness' of the generation.
                # (Omit this parameter or pass 0 to use a random seed.)
                "seed": int(seed),
                # How strongly the video sticks to the original image.
                # Use lower values to allow the model more freedom to make changes and higher values to correct motion distortions.
                "cfg_scale": float(cfg_scale),
                # Lower values generally result in less motion in the output video, while higher values generally result in more motion.
                # This parameter corresponds to the motion_bucket_id parameter from the paper.
                "motion_bucket_id": int(motion_bucket_id),
            }
            try:
                response = requests.post(
                    f"https://api.stability.ai/v2alpha/generation/image-to-video",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key_stability}",
                    },
                    files={"image": open(user_input, "rb")},
                    data=prepare_data,
                )
                gen_id_from_request = response.json().get("id")
                print(response)
                if response.status_code == 200:
                    content = "Success. Please pull video later."
                    save_generate_id_stabilityai(
                        gen_id_from_request,
                        int(seed),
                        float(cfg_scale),
                        int(motion_bucket_id),
                        user_input,
                    )
                    RowIdForSelect = get_generation_id_row(gen_id_from_request)
                    print(RowIdForSelect)
                    return (
                        jsonify(
                            content=content,
                            gen_id_from_request=gen_id_from_request,
                            RowIdForSelect=RowIdForSelect,
                        ),
                        200,
                    )
            except Exception as e:
                print(e)
                content = f"Oops something went wrong.: {response.text}"
                # raise Exception("Non-200 response: " + str(response.text))
            finally:
                is_retreived = show_generation_id()

    if request.method == "GET":
        if "generation_id" in request.args:
            generation_id = request.args.get("generation_id")
            db_gen_id = request.args.get("db_gen_id")
            print(generation_id)
            try:
                response = requests.request(
                    "GET",
                    f"https://api.stability.ai/v2alpha/generation/image-to-video/result/{generation_id}",
                    headers={
                        "Accept": "video/*",  # Use 'application/json' to receive base64 encoded JSON
                        "Authorization": f"Bearer {api_key_stability}",
                    },
                )
                print(response)
                if response.status_code == 404:
                    print(response)
                    delete_generate_id_stabilityai(db_gen_id)
                    return jsonify(content=response.status_code), 404
                if response.status_code == 202:
                    content = f"Generation in-progress, try again in 10 seconds."
                    return jsonify(content=response.status_code), 202
                # elif "Video/mp4" not in response.headers["Content-Type"]:
                #     print("note a video file")
                elif response.status_code == 200:
                    print("Generation complete!")
                    num = random.random()
                    print(response.headers)
                    with open(
                        f"./static/video/videoStabilityAI_{num}.mp4", "wb"
                    ) as file:
                        file.write(response.content)
                    content = f"videoStabilityAI_{num}.mp4"
                    update_generate_id_stabilityai(db_gen_id)
                    return jsonify(content=content), 200
            except Exception as e:
                print(e)
                content = f"Oops something went wrong.: {response.text}"
                # raise Exception("Non-200 response: " + str(response.text))
                return jsonify(content=content), 404
            # finally:
            #     return jsonify(content=content), 200

    return render_template(
        "imageToVideo_stabilityai.html", page="Image to Video", **locals()
    )


@app.route("/chatbotimage.html", methods=["POST", "GET"])
def chatbotimage():
    if request.method == "POST":
        user_input = request.form["prompt"]
        messages = []
        # sets the role of chatgpt
        # system: facts, basic and important context (no behaviour instructions here): example: At this place apples are purple.
        messages.append(
            {
                "role": "system",
                "content": "You are an expert in 3D animation, with extensive knowledge of rigging and animating 3D characters and objects. Your skill set includes proficiency in software such as Maya or Blender, as well as a strong understanding of character anatomy and movement principles.Rig and animate 3D characters and objects by first creating a skeleton or rig for each character or object. Use bones or joints to define the structure and movement of the model. Then, apply animation techniques such as keyframing or motion capture to bring the characters and objects to life. Use software like Maya or Blender to complete this task.",
            }
        )
        question = {}
        question["role"] = "user"
        question["content"] = user_input
        messages.append(question)
        try:
            # Generate images using DALL-E
            response = client.images.generate(
                model="dall-e-3",
                prompt=user_input,
                n=1,
                quality="standard",
                size="1024x1024",
            )
            # content = response.choices[0].message["content"]
            # Generate images using DALL-E
            content = response.data[0].url
            # count_tokens = response.usage.total_tokens
        except Exception as e:
            print(e)
            content = f"Oops something went wrong.: {e}"
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(e.http_status)
            print(e.error)
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            content = f"OpenAI API request exceeded rate limit: {e}. "
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.PermissionError:
            # Handle permission error, e.g. check scope or log
            content = f"OpenAI API request was not permitted: {e}"

        return jsonify(content=content), 200

    return render_template("chatbotimage.html", page="Chatbot project", **locals())


def save_response_token(token_val):
    try:
        create_dt = datetime.datetime.now()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT into value_token (token_val,create_dt) values (?,?)",
            (token_val, create_dt),
        )
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def save_generate_id_stabilityai(
    token_val, seed, cfg_scale, motion_bucket_id, image_path
):
    try:
        create_dt = datetime.datetime.now()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT into gen_id (token_val,create_dt,seed,cfg_scale,motion_bucket_id,image_path) values (?,?,?,?,?,?)",
            (token_val, create_dt, seed, cfg_scale, motion_bucket_id, image_path),
        )
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def update_generate_id_stabilityai(id):
    try:
        update_dt = datetime.datetime.now()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE gen_id SET is_retreived = ?, update_dt = ? WHERE id = ?",
            ["1", update_dt, id],
        )
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def delete_generate_id_stabilityai(id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE from gen_id WHERE id = ?", [id])
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def delete_not_stored_chromadb_collection(coll_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE from collections WHERE collection_name = ?", [coll_name])
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def save_system_instruction(content):
    try:
        create_dt = datetime.datetime.now()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT into system_instruction (content,create_dt) values (?,?)",
            (content, create_dt),
        )
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def store_chroma_in_sql_collection(
    collection_name,
    uuid,
    metadata,
    chunk_size,
    chunk_overlap,
    embed_model,
    persist_directory,
    url,
    pdf,
    semantic,
):
    try:
        create_dt = datetime.datetime.now()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT into collections (collection_name,uuid,metadata,create_dt,chunk_size,chunk_overlap,embed_model,persist_directory,url,pdf,semantic) values (?,?,?,?,?,?,?,?,?,?,?)",
            (
                collection_name,
                uuid,
                metadata,
                create_dt,
                chunk_size,
                chunk_overlap,
                embed_model,
                persist_directory,
                url,
                pdf,
                semantic,
            ),
        )
        conn.commit()
    except sqlite3.Error as er:
        print("SQLite error: %s" % (" ".join(er.args)))
        print("Exception class is: ", er.__class__)
        print("SQLite traceback: ")
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    except:
        conn.rollback()
        print("db error")
    finally:
        conn.close()
        print("closed!!!")


def show_system_instruction():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select * from system_instruction")
    rows = cur.fetchall()
    conn.close()
    return rows


def show_generation_id():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select * from gen_id where is_retreived='0'")
    rows = cur.fetchall()
    conn.close()
    return rows


def show_sql_collections():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select * from collections")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_generation_id_row(token_val):
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"select id from gen_id where token_val='{token_val}'")
    row = cur.fetchone()
    conn.close()
    return row[0]


def listInstalledModels():
    curl_command = f"curl http://localhost:11434/api/tags"

    output = subprocess.check_output(curl_command, shell=True, encoding="utf-8")
    res = json.loads(output)

    return res


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port='8888',debug=True)
    app.run(debug=True, threaded=True)
