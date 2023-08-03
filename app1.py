import gradio as gr
import requests
import ast
import logging
import io
import datetime
import random
import spacy
import re
from dateutil import parser as date_parser
import dateutil.parser

from datetime import timedelta
nlp = spacy.load('en_core_web_sm')
from datetime import date, timedelta
import requests
import re
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pytz

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle

import spacy

from bs4 import BeautifulSoup
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU



import json
from serpapi import GoogleSearch
from datetime import datetime, timedelta
import requests
from geotext import GeoText

from haystack.telemetry import tutorial_running

tutorial_running(21)
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer,TFBertModel
from sklearn.metrics import classification_report
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense 


df_test = pd.read_csv("C:/Users/HP/Desktop/maliha's code/test_da_10.csv", header = None, sep =',', names = ['text', 'plabel', 'label'],encoding='mac_roman', skiprows=[0])

df_test = df_test.dropna()

###################################### 45 for bert_new_and_old
import string
import unidecode

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    text = text.lower()
    return text

l = []

for i in df_test.text:
    i = remove_punctuations(i)
    l.append(i)
    
df_test.text = l


encoded_dict = {'a':0, 'd':1, 'f':2, 'g':3, 'i':4, 'pn':5, 's':6, 'yn':7}  #{'d':0, 'q':1, 's':2} {'d':0, 'r':1, 'yn':2, 'f':3, 'is':4, 'nis':5} {'d':0, 'i':1, 'yn':2, 'f':3, 's':4} 
df_test['label'] = df_test.label.map(encoded_dict)
pred_dict ={'apology':'a','direct order':'d','factual question':'f', 'greeting':'g','indirect order':'i','feedback':'pn','statement':'s','yes/no question':'yn'}
################################

y_test = to_categorical(df_test.label)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')#"bert-base-multilingual-uncased" bert-base-cased
bert = TFBertModel.from_pretrained('bert-base-cased')


max_len = 70

input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")#, padding=True)
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")#, padding=True)

embeddings = bert(input_ids,attention_mask = input_mask)[0] 
print(embeddings)
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(8,activation = 'sigmoid')(out)#3
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True #

### Model Compilation
optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

##########################################


checkpoint_path = "C:/Users/HP/Desktop/maliha's code/bert_checkponts_da/"

model.load_weights(checkpoint_path)
##########################################
##########################################
# Create a Stream to hold the log
log_stream = io.StringIO()

# Configure logging to write to that stream
logging.basicConfig(level=logging.INFO, stream=log_stream, format='%(message)s')
def alpaca_lora(instruction,input):
    url = 'http://129.128.243.13:25501/evaluate'

    data = {
        "instruction": instruction,
        "input": input,
        "temperature": 0.1,
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "max_new_tokens": 128,
    }

    response = requests.post(url, json=data)
   
    return response.json()['data']
def mpt30b(input):
    response = requests.post('http://129.128.243.13:25500/generate', json={'query': input
    })
    return response.json()['response']
    
def mpt30bclas (input):
    response = requests.post('http://129.128.243.13:25600/generate', json={'query': input
    })
    return response.json()['response']
# Mapping from weekday name to dateutil's weekday constant
WEEKDAY_MAPPING = {
    "monday": MO,
    "tuesday": TU,
    "wednesday": WE,
    "thursday": TH,
    "friday": FR,
    "saturday": SA,
    "sunday": SU,
}

# Special cases mapping
SPECIAL_CASES = {
    "today": 0,
    "tomorrow": 1,
    "next day": 2,
}

def get_date_from_sentence(sentence):
    # Lowercase the sentence to ensure we catch all instances
    sentence = sentence.lower()

    # Check for special cases
    for case in SPECIAL_CASES:
        if case in sentence:
            target_date = datetime.now() + timedelta(days=SPECIAL_CASES[case])
            return target_date.strftime(f"The date for {case} is %d/%m/%Y.")

    # Parse the day of the week from the sentence
    match = re.search(r"next (\w+)", sentence)
    if match:
        day_name = match.group(1)
    else:
        return "Couldn't find a day of the week in the sentence."

    # Check if the parsed day name is valid
    if day_name not in WEEKDAY_MAPPING:
        return "Invalid day of the week."

    # Get the current date
    now = datetime.now()

    # Get the weekday constant from the mapping
    weekday_constant = WEEKDAY_MAPPING[day_name]

    # Use dateutil's relativedelta to find the next specified day of the week
    next_day = now + relativedelta(weekday=weekday_constant)

    # Format the date as DD/MM/YYYY
    formatted_date = next_day.strftime("%d/%m/%Y")

    return f"The next {day_name.capitalize()} will be on {formatted_date}."
def predict_label(text):
    x_test = tokenizer(
    text= [text],
    add_special_tokens=True,
    padding="max_length",
    max_length=max_len, #25
    truncation=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
    x_test=dict(x_test)
    predicted_raw = model.predict(x_test)

    y_predicted = (np.argmax(predicted_raw, axis = 1)) #-1
    predicted_label = list(encoded_dict.keys())[list(encoded_dict.values()).index(y_predicted[0])]
    predicted = list(pred_dict.keys())[list(pred_dict.values()).index(predicted_label)]

    return predicted


SCOPES = ['https://www.googleapis.com/auth/calendar']

CREDENTIALS_FILE = 'credentials.json'
from datetime import datetime, timedelta

def get_date(day):
    # Map string inputs to number of days from today
    day_mapping = {
        "today": 0,
        "tomorrow": 1,
        "next day": 2,
        # Add more as needed...
    }
    
    if day in day_mapping:
        n = day_mapping[day]
    else:
        return "Invalid input. Please enter 'today', 'tomorrow', or 'next day'."

    # Get the current date and add 'n' days
    date_n_days_ahead = datetime.today() + timedelta(days=n)

    # Format the date
    formatted_date = date_n_days_ahead.strftime(f"{day} is %A, %B %d, %Y")

    return formatted_date

def detect_day1(sentence):
    # Lowercase the sentence to ensure we catch all instances
    sentence = sentence.lower()

    if "today" in sentence:
        return get_date("today")
    elif "tomorrow" in sentence:
        return get_date("tomorrow")
    elif "next day" in sentence:
        return get_date("next day")
    else:
        return "No specific day found in sentence."

def get_calendar_service():
   creds = None
   # The file token.pickle stores the user's access and refresh tokens, and is
   # created automatically when the authorization flow completes for the first
   # time.
   if os.path.exists('token.pickle'):
       with open('token.pickle', 'rb') as token:
           creds = pickle.load(token)
   # If there are no (valid) credentials available, let the user log in.
   if not creds or not creds.valid:
       if creds and creds.expired and creds.refresh_token:
           creds.refresh(Request())
       else:
           flow = InstalledAppFlow.from_client_secrets_file(
               CREDENTIALS_FILE, SCOPES)
           creds = flow.run_local_server(port=0)

       # Save the credentials for the next run
       with open('token.pickle', 'wb') as token:
           pickle.dump(creds, token)

   service = build('calendar', 'v3', credentials=creds)
   return service

def get_event(time):
    service = get_calendar_service()
    
    if service is not None:
        timeMax = time + timedelta(days=1)

        time = time.isoformat() +'Z'
        timeMax = timeMax.isoformat() + 'Z'

        events = service.events().list(
            calendarId='primary', 
            timeMin=time,
            timeMax = timeMax,
            maxResults=10, 
            singleEvents=True,
            orderBy='startTime'
        ).execute().get("items",[])

        if len(events) > 0 :
            print(events[0]["summary"])
            return events[0]["summary"]
        
        else : 
            answer = 'no event'
            return answer  
    else:
        print("Unable to establish connection with the service")
        return None
def add_event(event_name, time):
   # creates one hour event tomorrow 10 AM IST
   service = get_calendar_service()

    

#    d = datetime.now().date()
#    tomorrow = datetime(d.year, d.month, d.day, 10)+timedelta(days=1)
#    start = tomorrow.isoformat()
   end = (time + timedelta(hours=1)).isoformat()



   event_result = service.events().insert(calendarId='primary',
       body={
           "summary": event_name,
           "description": 'This is a tutorial example of automating google calendar with python',
           "start": {"dateTime": time.isoformat(), "timeZone": 'Africa/Tunis'},
           "end": {"dateTime": end, "timeZone": 'Africa/Tunis'},
       }
   ).execute()

   print("created event")
   print("id: ", event_result['id'])
   print("summary: ", event_result['summary'])
   print("starts at: ", event_result['start']['dateTime'])
   print("ends at: ", event_result['end']['dateTime'])


def predict_label(text):
    x_test = tokenizer(
    text= [text],
    add_special_tokens=True,
    padding="max_length",
    max_length=max_len, #25
    truncation=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
    x_test=dict(x_test)
    predicted_raw = model.predict(x_test)

    y_predicted = (np.argmax(predicted_raw, axis = 1)) #-1
    predicted_label = list(encoded_dict.keys())[list(encoded_dict.values()).index(y_predicted[0])]
    predicted = list(pred_dict.keys())[list(pred_dict.values()).index(predicted_label)]
    

    return predicted   
def get_answer_box(query):
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    answer_box = soup.find("div", class_="Z0LcW")
    
    if answer_box:
        return answer_box.get_text()
    else:
        return None
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
def detect_day(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    time_indicators = ['morning', 'afternoon', 'evening', 'night','tonight','today']
    for token in doc:
        if token.text.lower() == 'tomorrow':
            next_token = token.nbor()
            

        if token.text.lower() == 'next':
            next_token = token.nbor()
            if next_token.text.lower() == 'day':
                return 'Next day'		
            elif next_token.text.lower() in time_indicators:
                return f'Next {next_token.text.lower()}'
        if token.text.lower() == 'next':
            next_token = token.nbor()
            if next_token.text.lower() == 'week':
                return 'Next week'		
        if token.text.lower() in days_of_week:
            return token.text.capitalize()

    for token in doc:
        if token.text.lower() in days_of_week:
            return token.text.capitalize()  # Return the detected day of the week

        if token.text.lower() in ['today', 'tomorrow']:
            return token.text.capitalize()  # Return relative day reference

        if token.text.lower() == 'next' and token.head.text.lower() in days_of_week:
            next_day = token.head.text.capitalize()
            return f"Next {next_day}"  # Return "Next <day of the week>"

    return None  # Return None if no relevant day is detected
def detect_time(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    time_indicators = ['morning', 'afternoon', 'evening', 'night','tonight','now']

    for token in doc:
        if token.text.lower() in time_indicators:
            return token.text.capitalize()

        if token.ent_type_ == "TIME":
            return token.text.capitalize()  # Return the detected time

        # Check for specific time format, e.g., 2:30 PM
        if ":" in token.text and token.i + 2 < len(doc):
            next_token = doc[token.i + 1]
            if next_token.text.lower() == "pm" or next_token.text.lower() == "am":
                return f"{token.text} {next_token.text.upper()}"

    return None

def get_current_location():
    url = "http://ip-api.com/json"  # IP Geolocation API endpoint
    response = requests.get(url)
    data = response.json()
    
    if data["status"] == "success":
        return data["city"]
    else:
        return None
nlp = spacy.load("en_core_web_sm")

def detect_locations(text):
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  # GPE represents geopolitical entities, LOC represents cities
            locations.append(ent.text)
    return locations    
def detect_locations2(text):
  places = GeoText(text)
  cities = places.cities
  return cities
def weather(city, time, day):
    
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        city = city.replace(" ", "+")
        res = requests.get(
            f'https://www.google.com/search?q={city}+weather+{time}+{day}&oq={city}+weather+{time}+{day}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8&hl=en', headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        location = soup.select('#wob_loc')[0].getText().strip()
        time1 = soup.select('#wob_dts')[0].getText().strip()
        info = soup.select('#wob_dc')[0].getText().strip()
        weather = soup.select('#wob_tm')[0].getText().strip()
        precipitation = soup.select('#wob_pp')[0].getText().strip()
        humidity = soup.select('#wob_hm')[0].getText().strip()
        weather = weather + "C"
        return 'day: ' + day + ', time: ' + time + ', Location: ' + city + ', weather: ' + info + ', temperature: ' + weather + ', precipitation: ' + precipitation + ', humidity: ' + humidity
def get_weather(input) :
     if detect_day(input):
      day =detect_day(input)
     else :
      day = ''  
     if detect_time(input):
      time = detect_time(input)
     else :
      time = ''

     if detect_locations(input) : 
        city = detect_locations(input)
        
     elif detect_locations2(input) :
        city= detect_locations2(input)
     else :  
        city = get_current_location()
    
     return weather(''.join(city),time,day)     
def extract_keyword(sentence):
    words = sentence.split()
    for i, word in enumerate(words):
        if word.lower() in ["about", "concerning", "regarding"] and i < len(words) - 1:
            return words[i + 1]        
        
def get_location_and_time():
    # Get location data
    location_data = requests.get('http://ip-api.com/json').json()
    
    # Get timezone and city
    timezone = location_data['timezone']
    city = location_data['city']

    # Get current time in the timezone
    local_time = datetime.now(pytz.timezone(timezone))

    # Format the time
    formatted_time = local_time.strftime("%I:%M %p").lstrip("0")

    return f"The current time in {city} is {formatted_time}"  

def convert_time(time_str):
    # Handle 24-hour format input
    if 'h' in time_str:
        return time_str

    # Handle 12-hour format input
    try:
        dt = datetime.strptime(time_str, "%I %p")
        return dt.strftime("%Hh")
    except ValueError:
        try:
            dt = datetime.strptime(time_str, "%I%p")
            return dt.strftime("%Hh")
        except ValueError:
            return time_str

def convert_sentence(sentence):
    time_pattern = re.compile(r'(\d{1,2} ?[AP]M)', re.IGNORECASE)
    matches = time_pattern.findall(sentence)
    for match in matches:
        sentence = sentence.replace(match, convert_time(match))
    return sentence
       
ma_liste=[]    
class response :
    def __init__(self) :
        self.conversation = []
        self.conversation1 = []
        self.history = []
    
    dictionnaire=''  
    global k
    k = 0 
    global b
    b = 0
    def answer(self, message,dictionnaire,liste):
        liste=liste or []
        self.history = []
        user_message1 = f"{message}"
        user_message= alpaca_lora("Please ONLY CHECK if the following sentence has any spelling errors and don't change the format of any numbers or time references. here is the text and return the correct text:",user_message1)
        user_message= convert_sentence(user_message).lower()
        print(user_message)
        log_stream.truncate(0)  # clear previous logs
        log_stream.seek(0)
        self.dictionnaire = dictionnaire or ""
        if self.dictionnaire =="":
            self.dictionnaire= self.dictionnaire +' ' + user_message
            response = mpt30bclas(f"""Determine with high precision whether this text "{user_message}" is:

- Functionality Query (Example: 'What can you do?' or 'what tasks can you do?')
- Other inquiry (Example: 'Who won the world series last year?' or 'What's the capital of France?')

In each case, make sure to classify the incoming text as accurately as possible and return only the assigned category from the list.""").lower()
            print(response)
            

            if 'functionality query' in response :
                logging.info('Functionality Query')
                answer = """I'm here to assist you in many ways! Here are some of the tasks I can help with:

1 :Set reminders: If you need to remember an important task or event, I can help set a reminder for you.
2 :Tell the time or date: Not sure about the current time or date? Just ask me!
3 :Provide weather updates: I can give you the latest weather updates for your location or any other place you're curious about.
4 :Share recipes: If you're looking for cooking inspiration, I can provide recipes based on your preferences or ingredients you have on hand.
5 :Manage your grocery list: I can help you keep track of your grocery list. You can add items, remove them, or ask me what's on your list at any time.
6 :Provide information: If you have any questions or need information on a topic, feel free to ask. I'll do my best to provide a useful and accurate response.


Remember, I'm here to make your life easier, so don't hesitate to ask for help!"""
                self.conversation.append([user_message1, answer] )
                self.dictionnaire=''
                self.history.append((user_message, answer))
                return self.conversation, self.dictionnaire, log_stream.getvalue()
    
            else :
                logging.info("Other inquiry")
                intent=self.dictionnaire
                print(self.dictionnaire)
                  
                # Call the predict_label function to get the predicted label
                predicted_label = predict_label(self.dictionnaire)
                if predicted_label=='factual question'  :
                    logging.info("question")
                    
                    response = mpt30bclas(f"""Please carefully examine the provided text "{user_message}" and assign it to one of the following categories: 'factual question' or 'not factual question'.

    When assigning a category, ensure your classification is as precise as possible. Here are some guidelines to follow:

    'Factual Question': These are questions that require a definitive, objective answer, usually based on known facts or data. For example:
    'Who is the current president of the United States?'
    'What is the capital of Canada?'
    'When was the Declaration of Independence signed?'
    
    'Not Factual Question': These are questions that don't seek objective facts or data. They might be seeking opinions, asking for advice, or making subjective inquiries. For example:
    'What dish can i make ?'
    'Can you tell me a joke?'
    'What's your opinion on climate change?'
    Your task is to identify whether the question being asked requires a factual answer or not. It's essential that your classification is as accurate as possible.""").lower()
                    
                    print(response)
                    if 'not factual question' in response :
                        response1 = mpt30bclas(f"""Using high precision, categorize this text "{user_message}" into one of the following groups: 

- Joke request (Example: 'Can you share a funny joke?')
- Weather request (Example: 'What is the forecast in San Francisco next week?')
- Timing request (Example: 'what time is it right now?')
- Other (Any request that does not fit into the above categories.)

For each instance, provide the most accurate classification and return only the assigned category from the list.""")
                        response1= response1.lower()
                        if 'joke request' in response1 :
                            logging.info ('joke request')
                            
                            response = mpt30b(f"{user_message}")

                            self.conversation.append([user_message1,response])
                            self.dictionnaire =''

                            self.history.append((user_message, response))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                            

                            

                        elif 'weather request' in response1  :
                            logging.info('weather request')
                            resp=mpt30b(f'generate a weather sentence with all the details in the input with applying the right and the precise tense : {get_weather(user_message)}')
                            logging.info(get_weather(user_message))
                            self.conversation.append([ user_message1, resp] )
                            self.dictionnaire=''

                            self.history.append((user_message, resp))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                        elif 'timing request' in response1:
                                logging.info('Timing request')
                                response = mpt30bclas(f"""Please classify this text "{user_message}" into one of these categories accurately: 'Time Request' (e.g., 'What time is it now?' or 'What's the current time in London?'), or 'Day or Date Request' (e.g., 'What is today's date?' or 'What day of the week is it?'). The system should only return the category label""").lower()
                                if 'day or date request' in response:
                                    logging.info('day and date request')
                                    resp = detect_day1(user_message)
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                                    
                                else :
                                    logging.info('time request')
                                    resp = get_location_and_time()
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                        
                        
                        
                        
                        
                        else :
                            response = mpt30b(f"{user_message}")
                            self.conversation.append([user_message1,  response])
                            self.dictionnaire =''

                            self.history.append((user_message, response))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()

                                 
                            
                    
                    else:
                        logging.info('factual question')
                        alpaca=mpt30bclas(f"""Using high precision, categorize this text "{user_message}" into one of the following groups: 

- Weather request (Example: 'What is the forecast in San Francisco next week?')
- Timing request (Example: 'what time is it right now?')
- Recipe request (Example: 'Search for a vegan lasagna recipe.')

- Other (Any request that does not fit into the above categories.)

For each instance, provide the most accurate classification and return only the assigned category from the list.""")
                        alpaca=alpaca.lower()
                        logging.info(alpaca)
                        print(alpaca)
                        if 'weather request' in alpaca:
                            resp=mpt30b(f'generate a weather sentence with all the details in the input with applying the right and the precise tense : {get_weather(user_message)}')
                            logging.info(get_weather(user_message))
                            self.conversation.append([user_message1, resp] )
                            self.dictionnaire=''
                            self.history.append((user_message, resp))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                        elif 'timing request' in alpaca:
                                logging.info('Timing request')
                                response = mpt30bclas(f"""Please classify this text "{user_message}" into one of these categories accurately: 'Time Request' (e.g., 'What time is it now?' or 'What's the current time in London?'), or 'Day or Date Request' (e.g., 'What is today's date?' or 'What day of the week is it?'). The system should only return the category label""").lower()
                                if 'day or date request' in response:
                                    logging.info('day and date request')
                                    resp = detect_day1(user_message)
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                                    
                                else :
                                    logging.info('time request')
                                    resp = get_location_and_time()
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                        elif 'recipe request' in alpaca:
                            response3 = mpt30b(f"{user_message}")    
                            self.conversation.append([ user_message1, response3] )
                            self.dictionnaire =''

                            self.history.append((user_message, response3))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                        else:
                            answer_box = get_answer_box(user_message)

                            query = intent
                            api_key = "AIzaSyBiwYGmQbFDqGc35RS6S1JFz48EnvaQDAw"

                            resource = build("customsearch", 'v1', developerKey=api_key).cse()
                            result = resource.list(q=query, cx='b4c09627286a9479c').execute()
                            if answer_box== None :
                                if "items" in result :
                                    item = result ["items"][0]
                                    title = item.get("title", "")
                                    link = item.get("link", "")
                                    snippet = item.get("snippet", "")
                                    message1 = f"{snippet}"
                                    
                                        
                                    self.conversation.append([ user_message1, message1] )
                                    self.dictionnaire =''
                                    self.history.append((user_message, message1))
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()

                                    
                                else:
                                    
                                    response3 = mpt30b(f"{user_message}")    
                                    self.conversation.append([ user_message1, response3] )
                                    self.dictionnaire =''

                                    self.history.append((user_message, response3))
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                            

                            

                         
                         
                         
                         
                         
                         
                         
                         
                         
                        
                
                
                elif predicted_label =='yes/no question'   :
                    logging.info("yes/no question")
                    data=mpt30bclas(f"""Using high precision, categorize this text "{user_message}" into one of the following groups: 

- Weather request (Example: 'What is the forecast in San Francisco next week?')
- Timing request (Example: 'what time is it right now?')
- Recipe request (Example: 'Search for a vegan lasagna recipe.')

- Other (Any request that does not fit into the above categories.)

For each instance, provide the most accurate classification and return only the assigned category from the list.""")
                    data=data.lower()
                    if 'weather request' in data:
                            resp=alpaca_lora(f"generate a precise response with all the details in the input for the following yes or no question : {user_message}", get_weather(user_message))
                            logging.info(get_weather(user_message))
                            self.conversation.append([user_message1, resp] )
                            self.dictionnaire =''

                            self.history.append((user_message, resp))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                    elif 'timing request' in data:
                                logging.info('Timing request')
                                response = mpt30bclas(f"""Please classify this text "{user_message}" into one of these categories accurately: 'Time Request' (e.g., 'What time is it now?' or 'What's the current time in London?'), or 'Day or Date Request' (e.g., 'What is today's date?' or 'What day of the week is it?'). The system should only return the category label""").lower()
                                if 'day or date request' in response:
                                    logging.info('day and date request')
                                    resp = detect_day1(user_message)
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                                    
                                else :
                                    logging.info('time request')
                                    resp = get_location_and_time()
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                    elif 'recipe request' in alpaca:
                            response3 = mpt30b(f"{user_message}")    
                            self.conversation.append([ user_message1, response3] )
                            self.dictionnaire =''

                            self.history.append((user_message, response3))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                    else:
                            response = mpt30b(f"{user_message}")
                            self.conversation.append([user_message1,  response])
                            self.dictionnaire=''
                            self.history.append(user_message, response)
                            return self.conversation, self.dictionnaire, log_stream.getvalue()

                    
                elif predicted_label=='direct order' or predicted_label=='indirect order' :
                    logging.info('order')
                    response1 = mpt30bclas(f"""Using high precision, categorize this text {user_message} into one of the following groups: 

- Calendar request (Example: 'remind me to go shopping ')
- Recipe request (Example: 'Search for a vegan lasagna recipe.')
- Joke request (Example: 'Can you share a funny joke?')
- Weather request (Example: 'What is the forecast in San Francisco next week?')
- Grocery list (Example: 'Put apples on my grocery list.')
- Timing request (Example: 'what time is it right now?')
- Other (Any request that does not fit into the above categories.)

For each instance, provide the most accurate classification and return only the assigned category from the list.""")
                    data1 = response1
                    data1= data1.lower()
                    print(data1)

                    if 'joke request' in data1:
                        
                        logging.info('joke request')

                        response = mpt30b(f"{user_message}")

                        self.conversation.append([user_message1,response])
                        self.dictionnaire =''

                        self.history.append((user_message, response))
                        return self.conversation, self.dictionnaire, log_stream.getvalue()  
                                            
                    elif 'calendar request' in data1:
                        logging.info('calendar request')
                        pattern = r"(\w+):([\w\s']+),?"

                        response = alpaca_lora("detect the event and the time of the input : example : remind me to go fishing at 17h next monday: {Event:go fishing , Time:17 , day:next monday} example : remind me to go fishing next monday: {Event:go fishing , Time: , day:next monday} example : remind me to go fishing at 17h: {Event:go fishing , Time:17 , day:} example : i have an appointment with the doctor : {Event:an appointment with the doctor , Time: , day:} example : i have to go to shopping tomorrow : {Event:go shopping , Time: , day:tomorrow}",self.dictionnaire).lower()
                        nlp = spacy.load("en_core_web_sm")

                        data = response.lower()
                        s = data.strip('{}') # supprimer les accolades
                        splits = s.split(',') # diviser par virgule

                        dictionary_data = {}
                        for split in splits:
                            key, value = map(str.strip, split.split(':')) # diviser par deux points et supprimer les espaces supplémentaires
                            if value: # ajouter seulement la paire clé-valeur au dictionnaire si la valeur n'est pas vide
                                dictionary_data[key] = value
                        
                        nlp = spacy.load("en_core_web_sm")

                        def has_duration_before_event(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ in ["TIME", "DURATION"]:
                                    return True
                            return False

                        def get_duration_before_event(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ in ["TIME", "DURATION"]:
                                    duration = ent.text
                                    return parse_duration(duration)
                            return None

                        def parse_duration(duration_str):
                            tokens = duration_str.lower().split()
                            if len(tokens) == 2 and tokens[1] == "hours":
                                return datetime.timedelta(hours=int(tokens[0]))
                            if len(tokens) == 2 and tokens[1] == "minutes":
                                return datetime.timedelta(minutes=int(tokens[0]))
                            # Add more conditions here to handle more duration formats
                            return None

                        def has_next_date(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ == "DATE" and "next" in ent.text.lower():
                                    return True
                            return False
                        def get_next_day_date(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ == "DATE" and "next" in ent.text.lower():
                                    next_day = ent.text.split()[-1]  # Extract the day of the week from the entity text
                                    return get_next_day(next_day)  # Call the previous code to get the date of the next day
                            return None
                        def get_next_day(day):
                            today = datetime.datetime.today()
                            weekday_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
                            days_until_next_day = (weekday_map[day.lower()] - today.weekday()) % 7
                            next_day = today + datetime.timedelta(days=days_until_next_day)
                            return next_day.strftime("%Y-%m-%d")
                        def set_global_day_from_dict():
                            # Declare 'day' as global
                            global day

                            # Set 'day' to the value of 'day' in the dictionary
                            if 'day' in dictionary_data:
                                # Set 'day' to the value of 'day' in the dictionary
                                day = dictionary_data['day']
                            else:
                                day = None
                        
                        def set_global_time_from_dict():
                            # Declare 'time' as global
                            global time

                            # Set 'time' to the value of 'time' in the dictionary
                            if 'time' in dictionary_data:
                                # Set 'day' to the value of 'day' in the dictionary
                                time = dictionary_data['time']
                            else:
                                time = None
                        def set_reminder():
                            global reminder
                            reminder =get_duration_before_event(self.dictionnaire) 
                        def set_event():
                            global event
                            if 'event' in dictionary_data:
                                event = dictionary_data['event']
                            else:
                                event = None    
                        
                            
                        set_global_day_from_dict()   
                        set_global_time_from_dict() 
                        set_reminder()   
                        set_event()
                        if not day and not time  :
                            print("day and time")
                            data =alpaca_lora("ask a question to the user about the exact day and time to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not day :
                            print("day")
                            data =alpaca_lora("ask a question to the user about the exact day to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not time:
                            print('time')
                            data =alpaca_lora("ask a question to the user about the exact time to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not reminder:
                            print("reminder")
                            data = 'when do you want to be reminded'
                            self.history.append((message, data))
                            self.conversation.append([user_message1,data])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        else :
                            time_str = dictionary_data['Time'] + ":00:00"  # Add seconds to time

                            def subtract_duration_from_time(time_str, reminder):
                                # Parse the time string to a timedelta
                                time_td = timedelta(hours=int(time_str[:2]), minutes=int(time_str[3:5]), seconds=int(time_str[6:]))

                                # Parse the duration string to a timedelta
                                duration_td = timedelta(hours=int(duration_str[:2]), minutes=int(duration_str[3:5]), seconds=int(duration_str[6:]))

                                # Subtract the duration from the time
                                new_time_td = time_td - duration_td

                                # Format the new timedelta back to a time string
                                new_time_str = str(int(new_time_td.total_seconds() // 3600)).zfill(2) + ':' + str(int((new_time_td.total_seconds() % 3600) // 60)).zfill(2) + ':' + str(int(new_time_td.total_seconds() % 60)).zfill(2)

                                return new_time_str
                            day_str = dictionary_data['day']
                            today = datetime.datetime.today()
                            date_str = ''
                            if day_str.lower() == 'next week':
                                next_week = today + timedelta(days=7)
                                date_str = next_week.strftime('%Y-%m-%d')
                            if has_next_date(self.dictionnaire):
                                next_day_date = get_next_day_date(self.dictionnaire)
                                date_str = next_day_date
                            
                            elif day_str.lower() == 'tomorrow':
                                next_week = today + timedelta(days=1)
                                date_str = next_week.strftime('%Y-%m-%d')
                            elif day_str.lower() == 'today':
                                date_str = today.strftime('%Y-%m-%d')
                            datetime_str = date_str + ' ' + new_time_str
                            new_time = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

                            
                            answer = add_event(event,new_time)
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            self.dictionnaire=''
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    elif 'grocery list' in data1:
                        logging.info('Grocery')
                        a=[]
                        
                        url = 'http://129.128.243.13:25501/evaluate'

                        data = {
                            "instruction": "return only the grocery mentioned in the input in a list",
                            "input": user_message,
                            "temperature": 0.1,
                            "top_p": 0.75,
                            "top_k": 40,
                            "num_beams": 4,
                            "max_new_tokens": 128,
                        }

                        response = requests.post(url, json=data)
                        

                        data = response.json()['data'].lower()
                        logging.info (data)
                        def convert_string_to_list(string):
                            try:
                                return json.loads(string.replace("'", "\""))
                            except json.JSONDecodeError:
                                return []
                        def convert_string_to_list1(input_string):
                            # Removing the square brackets and splitting by comma to convert to list
                            items = input_string.strip('[]').split(', ')
                            return items    
                        def remove_apostrophes(my_list):
                            return [item.replace("'", "") for item in my_list]
                        a =convert_string_to_list(data[0]) or convert_string_to_list1(data)
                        a = remove_apostrophes(a)
                        
                        def add_list_without_duplicates(list2):
                            global ma_liste

                            seen = set(ma_liste)
                            for item in list2:
                                if item not in seen:
                                    ma_liste.append(item)
                                    seen.add(item)
                            
    
                        def supprimer_elements(elements):
                            global ma_liste
                            for element in elements:
                                if element in ma_liste:
                                    ma_liste.remove(element)    
                        if "add to grocery list" in mpt30bclas(f"""You are tasked with categorizing the given sentence "{user_message}". Determine whether the user is indicating a lack or surplus of an item. If the user's text implies a deficit (e.g., 'I'm out of [item]', 'I need more [item]', etc.), the correct categorization is 'Add to grocery list'. If the user's text implies a surplus (e.g., 'I have plenty of [item]', 'I don't need [item]', etc.), the categorization should be 'Remove from grocery list'.

For instance, if the user says 'I need more bananas', the response should be 'Add to grocery list'. If the user says 'I have too many potatoes', the response should be 'Remove from grocery list'. RETURN ONLY THE ASSIGNED CATEGORY FROM THE LIST""").lower() :
                            add_list_without_duplicates(a)
                            logging.info("add to the grocery list")
                            logging.info(ma_liste)
            
                        
                        
                            answer='the item is added and your shopping list contains :'+ " ".join(ma_liste)
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            self.dictionnaire=''
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                        else : 
                            logging.info ("remove from the list")
                            if len(ma_liste)<1:
                                answer=f"But you have nothing on the grocery list"
                                self.dictionnaire=''
                                self.history.append((message, answer))
                                self.conversation.append([user_message1,answer])
                                return self.conversation, self.dictionnaire, log_stream.getvalue()
                            else:
                                supprimer_elements(a)
                                logging.info(ma_liste)
                                answer='the item is removed and your shopping list contains :'+ " ".join(ma_liste)
                                self.history.append((message, answer))
                                self.conversation.append([user_message1,answer])
                                self.dictionnaire=''
                                return self.conversation, self.dictionnaire, log_stream.getvalue()
                                
                            
                    elif 'recipe request' in data1:
                        logging.info("recipe")
                        food = mpt30b(f"{user_message}")
                        query = food
                        
                        self.history.append((message, food))
                        self.conversation.append([user_message1,food])
                        self.dictionnaire=''
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
                    elif 'timing request' in data1:
                                logging.info('Timing request')
                                response = mpt30bclas(f"""Please classify this text "{user_message}" into one of these categories accurately: 'Time Request' (e.g., 'What time is it now?' or 'What's the current time in London?'), or 'Day or Date Request' (e.g., 'What is today's date?' or 'What day of the week is it?'). The system should only return the category label""").lower()
                                if 'day or date request' in response:
                                    logging.info('day and date request')
                                    resp = detect_day1(user_message)
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                                    
                                else :
                                    logging.info('time request')
                                    resp = get_location_and_time()
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                    elif 'weather request' in data1:
                            resp=mpt30b(f'generate a weather sentence with all the details in the input with applying the right and the precise tense : {get_weather(user_message)}')
                            logging.info(get_weather(user_message))
                            self.conversation.append([user_message1, resp] )
                            self.dictionnaire =''

                            self.history.append((user_message, resp))
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                     
                        
                    else : 
                        response = mpt30b(f"{user_message}") 
                        self.history.append((message, ''.join(response)))
                        self.conversation.append([user_message1,''.join(response)])
                        self.dictionnaire=''    
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
            
                                
                        
                
                    
                elif  predicted_label=='greeting':
                    logging.info('Greeting')
                    response=mpt30b(f"{user_message}")
                    data = response
                    logging.info("mpt30b response")
                    self.dictionnaire =''


                    self.conversation.append([user_message1, data ])
                    self.history.append((user_message))    
                    return self.conversation, self.dictionnaire ,log_stream.getvalue()
                    
                        

                elif  predicted_label=='statement' :
                    logging.info("statement")
                    alpaca =mpt30bclas(f"""Using high precision, categorize this text "{user_message}" into one of the following groups: 

- Calendar request (Example: 'remind me to go shopping ')
- Recipe request (Example: 'Search for a vegan lasagna recipe.')
- Joke request (Example: 'Can you share a funny joke?')
- Weather request (Example: 'What is the forecast in San Francisco next week?')
- Grocery list (Example: 'Put apples on my grocery list.')
- Timing request (Example: 'what time is it right now?')
- Other (Any request that does not fit into the above categories.)

For each instance, provide the most accurate classification and return only the assigned category from the list.""").lower()
                    print (alpaca)
                    if 'joke request' in alpaca :
                            
                        response = mpt30b(f"{user_message}")

                        self.conversation.append([user_message1,response])
                        self.dictionnaire =''

                        self.history.append((user_message, response))
                        return self.conversation, self.dictionnaire, log_stream.getvalue() 
                    
                    elif 'grocery list' in alpaca:
                        logging.info('Grocery')
                        a=[]
                        
                        url = 'http://129.128.243.13:25501/evaluate'

                        data = {
                            "instruction": "return only the grocery mentioned in the input in a list",
                            "input": self.dictionnaire,
                            "temperature": 0.1,
                            "top_p": 0.75,
                            "top_k": 40,
                            "num_beams": 4,
                            "max_new_tokens": 128,
                        }

                        response = requests.post(url, json=data)
                        

                        data = response.json()['data'].lower()
                        logging.info(data)
                        def convert_string_to_list(string):
                            try:
                                return json.loads(string.replace("'", "\""))
                            except json.JSONDecodeError:
                                return []
                        def convert_string_to_list1(input_string):
                            # Removing the square brackets and splitting by comma to convert to list
                            items = input_string.strip('[]').split(', ')
                            return items   



                        def remove_apostrophes(my_list):
                            return [item.replace("'", "") for item in my_list]
                        a =convert_string_to_list(data[0]) or convert_string_to_list1(data)
                        a = remove_apostrophes(a)                    
                        
                        def add_list_without_duplicates(list2):
                            global ma_liste

                            seen = set(ma_liste)
                            for item in list2:
                                if item not in seen:
                                    ma_liste.append(item)
                                    seen.add(item)   
                        def supprimer_elements(elements):
                            global ma_liste
                            for element in elements:
                                if element in ma_liste:
                                    ma_liste.remove(element)    
                        if "add to grocery list" in mpt30bclas(f"""You are tasked with categorizing the given sentence "{user_message}". Determine whether the user is indicating a lack or surplus of an item. If the user's text implies a deficit (e.g., 'I'm out of [item]', 'I need more [item]', etc.), the correct categorization is 'Add to grocery list'. If the user's text implies a surplus (e.g., 'I have plenty of [item]', 'I don't need [item]', etc.), the categorization should be 'Remove from grocery list'.

For instance, if the user says 'I need more bananas', the response should be 'Add to grocery list'. If the user says 'I have too many potatoes', the response should be 'Remove from grocery list'. RETURN ONLY THE ASSIGNED CATEGORY FROM THE LIST""").lower() :
                            logging.info("add to the grocery list")
                            logging.info(ma_liste)
            
                        
                        
                            answer=f"do you want me to add {data} to the grocery list?"
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                        else : 
                            logging.info ("remove from the list")
                            logging.info(ma_liste)
                            if len(ma_liste)<1:
                                answer=f"but you have nothing on the grocery list"
                                self.history.append((message, answer))
                                self.dictionnaire=''
                                self.conversation.append([user_message1,answer])
                                return self.conversation, self.dictionnaire, log_stream.getvalue()
                            else:
                                answer=f"do you want me to remove {data} from the grocery list?"
                                self.history.append((message, answer))
                                self.conversation.append([user_message1,answer])
                                return self.conversation, self.dictionnaire, log_stream.getvalue()
                                
        
                    elif 'recipe request' in alpaca:
                        logging.info("recipe")
                        food = mpt30b(f"{user_message}")
                        query = food
                        
                        self.history.append((message, food))
                        self.conversation.append([user_message1,food])
                        self.dictionnaire=''
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
                    elif 'calendar request' in alpaca:
                        logging.info('calendar request')
                        pattern = r"(\w+):([\w\s']+),?"

                        response = alpaca_lora("detect the event and the time of the input : example : remind me to go fishing at 17h next monday: {Event:go fishing , Time:17 , day:next monday} example : remind me to go fishing next monday: {Event:go fishing , Time: , day:next monday} example : remind me to go fishing at 17h: {Event:go fishing , Time:17 , day:} example : i have an appointment with the doctor : {Event:an appointment with the doctor , Time: , day:} example : i have to go to shopping tomorrow : {Event:go shopping , Time: , day:tomorrow}",self.dictionnaire).lower()
                        nlp = spacy.load("en_core_web_sm")

                        data = response.lower()
                        s = data.strip('{}') # supprimer les accolades
                        splits = s.split(',') # diviser par virgule

                        dictionary_data = {}
                        for split in splits:
                            key, value = map(str.strip, split.split(':')) # diviser par deux points et supprimer les espaces supplémentaires
                            if value: # ajouter seulement la paire clé-valeur au dictionnaire si la valeur n'est pas vide
                                dictionary_data[key] = value
                        
                        nlp = spacy.load("en_core_web_sm")

                        def has_duration_before_event(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ in ["TIME", "DURATION"]:
                                    return True
                            return False

                        def get_duration_before_event(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ in ["TIME", "DURATION"]:
                                    duration = ent.text
                                    return parse_duration(duration)
                            return None

                        def parse_duration(duration_str):
                            tokens = duration_str.lower().split()
                            if len(tokens) == 2 and tokens[1] == "hours":
                                return datetime.timedelta(hours=int(tokens[0]))
                            if len(tokens) == 2 and tokens[1] == "minutes":
                                return datetime.timedelta(minutes=int(tokens[0]))
                            # Add more conditions here to handle more duration formats
                            return None

                        def has_next_date(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ == "DATE" and "next" in ent.text.lower():
                                    return True
                            return False
                        def get_next_day_date(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ == "DATE" and "next" in ent.text.lower():
                                    next_day = ent.text.split()[-1]  # Extract the day of the week from the entity text
                                    return get_next_day(next_day)  # Call the previous code to get the date of the next day
                            return None
                        def get_next_day(day):
                            today = datetime.datetime.today()
                            weekday_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
                            days_until_next_day = (weekday_map[day.lower()] - today.weekday()) % 7
                            next_day = today + datetime.timedelta(days=days_until_next_day)
                            return next_day.strftime("%Y-%m-%d")
                        def set_global_day_from_dict():
                            # Declare 'day' as global
                            global day

                            # Set 'day' to the value of 'day' in the dictionary
                            if 'day' in dictionary_data:
                                # Set 'day' to the value of 'day' in the dictionary
                                day = dictionary_data['day']
                            else:
                                day = None
                        
                        def set_global_time_from_dict():
                            # Declare 'time' as global
                            global time

                            # Set 'time' to the value of 'time' in the dictionary
                            if 'time' in dictionary_data:
                                # Set 'day' to the value of 'day' in the dictionary
                                time = dictionary_data['time']
                            else:
                                time = None
                        def set_reminder():
                            global reminder
                            reminder =get_duration_before_event(self.dictionnaire) 
                        def set_event():
                            global event
                            if 'event' in dictionary_data:
                                event = dictionary_data['event']
                            else:
                                event = None    
                        
                            
                        set_global_day_from_dict()   
                        set_global_time_from_dict() 
                        set_reminder()   
                        set_event()
                        if not day and not time  :
                            print("day and time")
                            data =alpaca_lora("ask a question to the user about the exact day and time to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not day :
                            print("day")
                            data =alpaca_lora("ask a question to the user about the exact day to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not time:
                            print('time')
                            data =alpaca_lora("ask a question to the user about the exact time to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not reminder:
                            print("reminder")
                            data = 'would you like to be reminded before the event?'
                            self.history.append((message, data))
                            self.conversation.append([user_message1,data])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        else :
                            time_str = dictionary_data['Time'] + ":00:00"  # Add seconds to time

                            def subtract_duration_from_time(time_str, reminder):
                                # Parse the time string to a timedelta
                                time_td = timedelta(hours=int(time_str[:2]), minutes=int(time_str[3:5]), seconds=int(time_str[6:]))

                                # Parse the duration string to a timedelta
                                duration_td = timedelta(hours=int(duration_str[:2]), minutes=int(duration_str[3:5]), seconds=int(duration_str[6:]))

                                # Subtract the duration from the time
                                new_time_td = time_td - duration_td

                                # Format the new timedelta back to a time string
                                new_time_str = str(int(new_time_td.total_seconds() // 3600)).zfill(2) + ':' + str(int((new_time_td.total_seconds() % 3600) // 60)).zfill(2) + ':' + str(int(new_time_td.total_seconds() % 60)).zfill(2)

                                return new_time_str
                            day_str = dictionary_data['day']
                            today = datetime.datetime.today()
                            date_str = ''
                            if day_str.lower() == 'next week':
                                next_week = today + timedelta(days=7)
                                date_str = next_week.strftime('%Y-%m-%d')
                            if has_next_date(self.dictionnaire):
                                next_day_date = get_next_day_date(self.dictionnaire)
                                date_str = next_day_date
                            
                            elif day_str.lower() == 'tomorrow':
                                next_week = today + timedelta(days=1)
                                date_str = next_week.strftime('%Y-%m-%d')
                            elif day_str.lower() == 'today':
                                date_str = today.strftime('%Y-%m-%d')
                            datetime_str = date_str + ' ' + new_time_str
                            new_time = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

                            
                            add_event(event,new_time)
                            answer='Event Added'
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            self.dictionnaire=''
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
                    
                    
                    
                    
                    
                    else :
                        answer = mpt30b(f"{user_message}")
                        logging.info('mpt30b response')
                        self.history.append((message, answer))
                        self.conversation.append([user_message1,answer])
                        self.dictionnaire=''
                        return self.conversation, self.dictionnaire, log_stream.getvalue()         
                        
                elif predicted_label=='apology' or predicted_label=='feedback': 
                    answer = mpt30b(f"""generate a response based on the input : {user_message}""")
                    self.history.append((message, answer))
                    self.conversation.append([user_message1,answer])
                    self.dictionnaire=''
                    return self.conversation, self.dictionnaire  , log_stream.getvalue()  
            
        else :
            print('else')
            response1 = mpt30b(f"""Using high precision, categorize this text "{user_message}" into one of the following groups: 

- Calendar request (Example: 'remind me to go shopping ')
- Recipe request (Example: 'Search for a vegan lasagna recipe.')
- Joke request (Example: 'Can you share a funny joke?')
- Weather request (Example: 'What is the forecast in San Francisco next week?')
- Grocery list (Example: 'Put apples on my grocery list.')
- Timing request (Example: 'what time is it right now?')
- Other (Any request that does not fit into the above categories.)

For each instance, provide the most accurate classification and return only the assigned category from the list.""")
            data1 = response1
            
            data1= data1.lower()
            logging.info(data1)  
            global k    
            def set_k(st):
                global k
                if st == "1":
                    k = 1
                else:
                    k = 0
            def set_k_empty():
                global k 
                k=0  
            def set_b(st):
                global b
                if st =='1':
                    b = 1
                else :
                    b= 0    
            def set_b_empty():
                global b
                b=0      
            def check_k():
                global k 
                if k==1 :
                    return True
                else :
                    return False
            def check_b():
                global b 
                if b==1 :
                    return True
                else :
                    return False
            def set_globals_empty():
                global day
                global time
                global reminder
                global event
                day = ''
                time = ''
                reminder = ''
                event =''    
            if 'calendar request' in data1:
                    print('calendar')
                    logging.info('calendar request')
                    data= mpt30bclas(f"""Using high precision, categorize this text "{user_message}" into one of the following groups: 

- Joke request (Example: 'Can you share a funny joke?')
- Timing request (Example: 'what time is it right now?')
- Other (Example: 'tomorrow at 5pm')

For each instance, provide the most accurate classification and return only the assigned category from the list.""")
                    data = data.lower()
                    print(data)
                    if k==1:
                        resp = alpaca_lora("""analyze the input to determine whether the sentiment expressed is one of acceptance (agreeing or saying "yes") or rejection (disagreeing or saying "no")""", user_message).lower()

                        if resp == 'acceptance':
                            set_b('1')
                            
                                
                        else :
                            answer= f"Absolutely, we can move on to something else. How may I assist you further? Do you have any other questions or is there another topic you'd like to discuss?"
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            self.dictionnaire =''
                            set_globals_empty()

                            set_k_empty()
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                        
                    if 'other' in data or b==1:
                        
                        pattern = r"(\w+):([\w\s']+),?"

                        nlp = spacy.load("en_core_web_sm")

                        

                        dictionary_data = {}
                        def get_day(text) :
                            global day
                            # Find day using regular expressions
                            day_search = re.search(r'(today|tomorrow|next weekend)', text)
                            
                            # If day was found, parse it and set the global variable
                            if day_search:
                                day = day_search.group(1)
                                
                        def get_time(text) :
                            global time
                            # Find time using regular expressions
                            time_search = re.search(r'(\d{1,2})h', text)
                            
                            # If time was found, parse it and set the global variable
                            if time_search:
                                time = int(time_search.group(1))
                        
                        
                        
                        
                        def has_duration_before_event(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ in ["TIME", "DURATION"]:
                                    return True
                            return False

                        def get_duration_before_event(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ in ["TIME", "DURATION"]:
                                    duration = ent.text
                                    return parse_duration(duration)
                            return None

                        def parse_duration(duration_str):
                            tokens = duration_str.lower().split()
                            if len(tokens) == 2 and tokens[1] == "hours":
                                return timedelta(hours=int(tokens[0]))
                            if len(tokens) == 2 and tokens[1] == "minutes":
                                return timedelta(minutes=int(tokens[0]))
                            # Add more conditions here to handle more duration formats
                            return None
                        def format_time(time_string):
                            hours, minutes, seconds = time_string.split(':')
                            return ':'.join([hours.zfill(2), minutes, seconds])
                                            
                        def set_reminder():
                            global reminder
                            if get_duration_before_event(user_message):
                                reminder = format_time((str(get_duration_before_event(user_message))))
                        
                        def has_next_date(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ == "DATE" and "next" in ent.text.lower():
                                    return True
                            return False
                        def get_next_day_date(text):
                            doc = nlp(text)
                            for ent in doc.ents:
                                if ent.label_ == "DATE" and "next" in ent.text.lower():
                                    next_day = ent.text.split()[-1]  # Extract the day of the week from the entity text
                                    return get_next_day(next_day)  # Call the previous code to get the date of the next day
                            return None
                        def get_next_day(day):
                            today = datetime.datetime.today()
                            weekday_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
                            days_until_next_day = (weekday_map[day.lower()] - today.weekday()) % 7
                            next_day = today + datetime.timedelta(days=days_until_next_day)
                            return next_day.strftime("%Y-%m-%d")
                        def set_globals_empty():
                            global day
                            global time
                            global reminder
                            global event
                            day = ''
                            time = ''
                            reminder = ''
                            event =''
                        
                        
                        get_day(user_message)   
                        get_time(user_message) 
                        set_reminder()   
                        if not day and not time  :
                            print("day and time")
                            data =f"sure ! i will put the {event} on the calendar! but can you provide me some more informations like the exact day and time?"
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not day :
                            print("day")
                            data =alpaca_lora("ask a question to the user about the exact day to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not time :
                            print('time')
                            data =alpaca_lora("ask a question to the user about the exact time to put in the calendar",self.dictionnaire)
                            self.history.append((message, ''.join(data)))
                            self.conversation.append([user_message1,''.join(data)])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        elif not reminder:
                            print("reminder")
                            data = 'Would you like to be reminded before the event?'
                            self.history.append((message, data))
                            self.conversation.append([user_message1,data])
                            return self.conversation,  self.dictionnaire, log_stream.getvalue()
                        
                        else :
                            time_str = str(time) + ":00:00"  # Add seconds to time
                            
                            def subtract_duration_from_time(time_str, reminder):
                                # Parse the time string to a timedelta
                                time_td = timedelta(hours=int(time_str[:2]), minutes=int(time_str[3:5]), seconds=int(time_str[6:]))

                                # Parse the duration string to a timedelta
                                duration_td = timedelta(hours=int(reminder[:2]), minutes=int(reminder[3:5]), seconds=int(reminder[6:]))

                                # Subtract the duration from the time
                                new_time_td = time_td - duration_td

                                # Format the new timedelta back to a time string
                                new_time_str = str(int(new_time_td.total_seconds() // 3600)).zfill(2) + ':' + str(int((new_time_td.total_seconds() % 3600) // 60)).zfill(2) + ':' + str(int(new_time_td.total_seconds() % 60)).zfill(2)

                                return new_time_str
                            new_time_str = subtract_duration_from_time(time_str, reminder)
                            day_str = str(day)
                            today = datetime.today()
                            date_str = ''
                            if day_str.lower() == 'next week':
                                next_week = today + timedelta(days=7)
                                date_str = next_week.strftime('%Y-%m-%d')
                            if has_next_date(self.dictionnaire):
                                next_day_date = get_next_day_date(self.dictionnaire)
                                date_str = next_day_date
                            
                            elif day_str.lower() == 'tomorrow':
                                next_week = today + timedelta(days=1)
                                date_str = next_week.strftime('%Y-%m-%d')
                            elif day_str.lower() == 'today':
                                date_str = today.strftime('%Y-%m-%d')
                            datetime_str = date_str + ' ' + new_time_str
                            new_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

                            
                            add_event(event,new_time)
                            answer = f"The event {event} is set in the calendar on {date_str} at {time_str} with success and you will be reminded at {new_time_str}."

                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            set_globals_empty()
                            set_k_empty()
                            set_b_empty()
                            self.dictionnaire=''
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                    elif 'joke request' in data :
                        logging.info ('joke request')
                        
                        response = mpt30b(f"user_message")
                        answer=f"Just a gentle reminder that we were setting up something about this event : {event}. Shall we finish that up?"
                        resp=response +'\n'+'\n'+'\n'+ answer
                        self.conversation.append([user_message1,resp])
                        set_k('1')
                        self.history.append((user_message, response))
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
                    elif 'timing request' in data:
                                logging.info('Timing request')
                                response = mpt30bclas(f"""Please classify this text "{user_message}" into one of these categories accurately: 'Time Request' (e.g., 'What time is it now?' or 'What's the current time in London?'), or 'Day or Date Request' (e.g., 'What is today's date?' or 'What day of the week is it?'). The system should only return the category label""").lower()
                                if 'day or date request' in response:
                                    logging.info('day and date request')
                                    resp = detect_day1(user_message)
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                                    
                                else :
                                    logging.info('time request')
                                    resp = get_location_and_time()
                                    self.history.append((message, resp))
                                    self.conversation.append([user_message1,resp])
                                    self.dictionnaire=''
                                    return self.conversation, self.dictionnaire, log_stream.getvalue()
                    
            elif 'grocery list' in data1:
                logging.info('Grocery')
                a=[]
                
                url = 'http://129.128.243.13:25501/evaluate'

                data = {
                    "instruction": "return only the grocery mentioned in the input in a list",
                    "input": self.dictionnaire,
                    "temperature": 0.1,
                    "top_p": 0.75,
                    "top_k": 40,
                    "num_beams": 4,
                    "max_new_tokens": 128,
                }

                response = requests.post(url, json=data)
                

                data = response.json()['data'].lower()
                logging.info(data)
                def convert_string_to_list(string):
                    try:
                        return json.loads(string.replace("'", "\""))
                    except json.JSONDecodeError:
                        return []
                def convert_string_to_list1(input_string):
                    # Removing the square brackets and splitting by comma to convert to list
                    items = input_string.strip('[]').split(', ')
                    return items   



                a =convert_string_to_list(data) or convert_string_to_list1(data)
                def add_list_without_duplicates(list2):
                    global ma_liste

                    seen = set(ma_liste)
                    for item in list2:
                        if item not in seen:
                            ma_liste.append(item)
                            seen.add(item)   
                def supprimer_elements(elements):
                    global ma_liste
                    for element in elements:
                        if element in ma_liste:
                            ma_liste.remove(element)    
                if "add to grocery list" in mpt30bclas(f"""You are tasked with categorizing the given sentence "{user_message}". Determine whether the user is indicating a lack or surplus of an item. If the user's text implies a deficit (e.g., 'I'm out of [item]', 'I need more [item]', etc.), the correct categorization is 'Add to grocery list'. If the user's text implies a surplus (e.g., 'I have plenty of [item]', 'I don't need [item]', etc.), the categorization should be 'Remove from grocery list'.

For instance, if the user says 'I need more bananas', the response should be 'Add to grocery list'. If the user says 'I have too many potatoes', the response should be 'Remove from grocery list'. RETURN ONLY THE ASSIGNED CATEGORY FROM THE LIST""").lower() :
                    logging.info("add to the grocery list")
                    logging.info(ma_liste)
                    resp = alpaca_lora("""analyze the input to determine whether the sentiment expressed is one of acceptance (agreeing or saying "yes") or rejection (disagreeing or saying "no")""", user_message).lower()
                    if resp == 'acceptance':
                        add_list_without_duplicates(a)
                        logging.info("acceptance to add")
                        answer= f"Great, I've added {data} to your grocery list and your list contains : {ma_liste}"
                        self.history.append((message, answer))
                        self.conversation.append([user_message1,answer])
                        self.dictionnaire =''
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
                    else :
                        answer =f"Understood, I will not add {data} to your grocery list."
                        logging.info("rejection to add")

                        self.history.append((message, answer))
                        self.conversation.append([user_message1,answer])
                        self.dictionnaire =''
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
                 
                
                    
            
                else : 
                    logging.info ("remove from the list")
                    resp = alpaca_lora("""analyze the input to determine whether the sentiment expressed is one of acceptance (agreeing or saying "yes") or rejection (disagreeing or saying "no")""", user_message).lower()

                    if resp == 'acceptance':
                        supprimer_elements(a)
                        if len(ma_liste)<1 :
                            answer= f"Done! You won't see {a} in your list anymore. Your list contains nothing"
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            self.dictionnaire =''
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                        else :
                            answer= f"Done! You won't see {a} in your list anymore. Your list contains :{ma_liste}"
                            self.history.append((message, answer))
                            self.conversation.append([user_message1,answer])
                            self.dictionnaire =''
                            return self.conversation, self.dictionnaire, log_stream.getvalue()
                            
                    else :
                        answer =f"Understood, I will keep it on your grocery list."
                        self.history.append((message, answer))
                        self.conversation.append([user_message1,answer])
                        self.dictionnaire =''
                        return self.conversation, self.dictionnaire, log_stream.getvalue()
                 
                    
                    
                
                

# User input
block = gr.Blocks()
calendar = response()

with block:
    gr.Markdown("""<h1><center>🤖 ANA-Assistant 🐍</center></h1>
                   <p><center>ANA-Assistant is a chatbot that uses the Alpaca Lora model</center></p>
    """)
   
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            message = gr.Textbox(label="Message", placeholder="Hi, how are things ?")
            state = gr.State()
            submit = gr.Button("Send", size="sm")
            
        with gr.Column(scale=1):
            log = gr.Textbox(label="Log", disabled=True)  # Textbox for displaying log information
    submit.click(calendar.answer, inputs=[message, state], outputs=[chatbot, state, log])
    submit.click(lambda x: gr.update(value=""), [state], [message], queue=False)
    # Textbox for displaying log information

    
  
   
block.launch(debug=True) 