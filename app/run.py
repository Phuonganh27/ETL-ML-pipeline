import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re 
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and preprocess a given text.
    This function performs the following tasks:
    1. Converts the text to lowercase.
    2. Removes non-alphanumeric characters and replaces them with spaces.
    3. Tokenizes the text into words.
    4. Removes punctuation.
    5. Removes common English stopwords.
    6. Lemmatizes the tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of preprocessed and tokenized words.
    """
    # Tokenize the text into words
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    tokens = word_tokenize(text)

    # Remove punctuation 
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = []
    categories = df.columns.difference(['id', 'message', 'original', 'genre'])
    for col in categories:
        counts = df.groupby(col).count()['message']
        names = ['yes', 'no']
        graph_object = {
            'data': [
                Bar(
                    x=names,
                    y=counts
                )
            ],

            'layout': {
                'title': f'Distribution of {col}',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': f"{col}"
                }
            }
        }
        graphs.append(graph_object)
    
    # Groupby the genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graph_object =   {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    graphs.append(graph_object)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()