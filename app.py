import os

import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import re

from bs4 import BeautifulSoup

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

dataset = pd.read_csv('https://raw.githubusercontent.com/Group-7-Big-Data/Assignment-2/master/IMDB_review_cleaned.csv')

app.layout = html.Div([
    html.H2('Movie review sentiment predictor:'),
    html.P('Please write down a sentence'),
    dcc.Input(id='sentence', value='', type='text', size='100'),
    html.P(''),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.P(''),
    html.P('(Warning: This is slow due to heroku free version)'),
    html.P('Prediction:'),
    html.P(''),
    html.Div(id='display-value')
])

#load_tfidf = pd.read_pickle('https://www.dropbox.com/s/xda1lt9jzxkmw3t/tfidf.sav?dl=1')

#load_svc = pd.read_pickle('https://www.dropbox.com/s/lcz7w8io99xggus/linearsvc.sav?dl=1')

def predict_sentence(text, tfidf, svc, df):
    text = BeautifulSoup(text).get_text()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [text]
    
    tfid_transformed_text = tfidf.transform(text)
    
    y_pred = svc.predict(tfid_transformed_text)
    
    return y_pred

load_tfidf = pd.read_pickle('https://www.dropbox.com/s/xda1lt9jzxkmw3t/tfidf.sav?dl=1')
load_svc = pd.read_pickle('https://www.dropbox.com/s/lcz7w8io99xggus/linearsvc.sav?dl=1')

@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('submit-button-state', 'n_clicks')],
              [dash.dependencies.State('sentence', 'value')])    
def display_value(n_clicks, value):
    output = []
    
    if n_clicks is not 0:        
        prediction = predict_sentence(value, load_tfidf, load_svc, dataset)
        output.append(html.P('Your sentence sentiment prediction is "{}"'.format(prediction[0])))
        output.append(html.P('Sentence: {}'.format(value)))
    
    return output
    

if __name__ == '__main__':
    app.run_server(debug=True)
