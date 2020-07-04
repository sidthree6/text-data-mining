import os

import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wordcloud

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

def predict_sentence(text, df):
    text = BeautifulSoup(text).get_text()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [text]
    
    extra = pd.Series(text)
    review_series = df.review.append(extra, ignore_index=True)
    
    tfid = TfidfVectorizer(ngram_range=(1,2))
    tfid_transformed = tfid.fit_transform(review_series)
    
    tfid_matrix = tfid_transformed[:50000]
    tfid_predict = tfid_transformed[-1:]
    
    X_train, X_test, y_train, y_test = train_test_split(tfid_matrix, df.sentiment, test_size=0.3, random_state=0)
    
    svc_m = LinearSVC()
    svc_m.fit(tfid_matrix, df.sentiment)
    
    y_pred = svc_m.predict(tfid_predict)
    
    return y_pred

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

@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('submit-button-state', 'n_clicks')],
              [dash.dependencies.State('sentence', 'value')])
def display_value(n_clicks, value):
    prediction = predict_sentence(value, dataset)
    output = []
    output.append(html.P('Your sentence sentiment prediction is "{}"'.format(prediction[0])))
    output.append(html.P('Sentence: {}'.format(value)))
    
    return output
    

if __name__ == '__main__':
    app.run_server(debug=True)