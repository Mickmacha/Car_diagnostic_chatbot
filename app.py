import pickle
import gradio as gr
import json
import random
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Load the trained model from the pickle file
with open('car-diagnostic.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the words and classes used to train the model
with open("intents.json") as file:
    data = json.load(file)
words = []
classes = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [stemmer.stem(w.lower()) for w in words if w not in ['?']]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Define functions for cleaning up sentences and creating a bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    bag = [0]*len(words)  
    for s in clean_up_sentence(sentence):
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(pd.DataFrame([bag], dtype=float, index=['input']))

# Use the model to classify new sentences


def classify(sentence):
    # Generate probabilities from the model
    input_data = bow(sentence, words)
    results = model.predict([input_data])[0]
    
    # Filter out predictions below a threshold, and provide intent index
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Create a list of predicted intents and their probabilities
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    
    # Choose a response at random for the top predicted intent
    if return_list:
        intent = return_list[0][0]  # Get the top predicted intent
        responses = None
        for intent_json in data['intents']:
            if intent_json['tag'] == intent:
                responses = intent_json['responses']
                break
        if responses:
            response = random.choice(responses)
            return {'intent': intent, 'probability': return_list[0][1], 'response': response}
    return None
    
# # Test the model 
# classify("What is the problem with my car?")       

def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=classify, inputs="text", outputs="label", title="Car Diagnostics", description="Chat with the bot")
iface.launch()