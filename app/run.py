import os
import re
import json
import string
import pandas as pd
import joblib
import plotly
from sqlalchemy import create_engine
from plotly.graph_objs import Bar
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from flask import Flask, jsonify, render_template, request

# Importing train_classifier
from models import train_classifier

app = Flask(__name__)


def tokenize(text):
    """Normalize, tokenize, and lemmatize text input, replacing URLs.

    Args:
        text (str): Input text.

    Returns:
        list: Processed tokens.
    """
    url_regex = r"(?:(?:https?|ftp):\/\/)?[\w\/\-?=%.]+\.[\w\/\-&?=%.]+"  # More robust URL regex
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove punctuation using re.sub
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token, tag in pos_tag(tokens):
        pos = 'n' if tag.startswith(
            "NN") else 'v' if tag.startswith('VB') else 'a'
        try:
            clean_token = lemmatizer.lemmatize(token, pos=pos).lower().strip()
            clean_tokens.append(clean_token)
        except Exception as e:
            continue  # Skip tokens that cause errors
    return clean_tokens


# Load data
try:
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
except Exception as e:
    print(f"Error loading database: {e}")
    df = None

# Load model
try:
    model = joblib.load("../models/classifier.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/')
@app.route('/index')
def index():
    if df is None:
        return "Error: Database not loaded properly."

    # Extract data needed for visuals
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_counts.index

    genre_counts = df.groupby('genre').count()['message']
    genre_names = genre_counts.index

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Categories", 'tickangle': -45}
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    if model is None:
        return "Error: Model not loaded properly."

    # Save user input in query
    query = request.args.get('query', '')

    if not query.strip():
        return "Error: Query cannot be empty."

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
