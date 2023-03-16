import pickle 
import random
import numpy

import nltk
nltk.download('punkt')
import json
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# load model
model = pickle.load(open("data.pickle", "rb"))

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
with open("intents.json") as file:
    data = json.load(file)

try: 
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

def chat(message):
    # print("Start talking with the bot (type quit to stop)!")
    # while True:
    #     inp = input("You: ")
    #     if inp.lower() == "quit":
    #         break
        with open("intents.json") as file:
          data = json.load(file)

        try: 
            with open("data.pickle", "rb") as f:
                words, labels, training, output = pickle.load(f)
        except:
            words = []
            labels = []
            docs_x = []
            docs_y = []
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        results = model.predict([bag_of_words(message, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return str(random.choice(responses))

    
chat("fuel issue")