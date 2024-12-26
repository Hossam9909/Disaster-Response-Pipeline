import sys
import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load data from SQLite database.
    Args:
    - database_filepath (str): Filepath for the SQLite database.
    Returns:
    - X (DataFrame): Features (messages).
    - Y (DataFrame): Targets (categories).
    - category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


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

    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token, tag in pos_tag(tokens):
        if tag.startswith("NN"):
            pos = 'n'  # Noun
        elif tag.startswith('VB'):
            pos = 'v'  # Verb
        else:
            pos = 'a'  # Adjective
        try:
            clean_token = lemmatizer.lemmatize(token, pos=pos).lower().strip()
            clean_tokens.append(clean_token)
        except:
            continue  # Handle cases where lemmatization might fail
    return clean_tokens


def build_model():
    """
    Build machine learning pipeline with GridSearchCV.
    Returns:
    - cv (GridSearchCV): Grid search model pipeline.
    """
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance on the test data.
    Args:
    - model: Trained machine learning model.
    - X_test (DataFrame): Test features.
    - Y_test (DataFrame): Test targets.
    - category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}\n', classification_report(
            Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.
    Args:
    - model: Trained machine learning model.
    - model_filepath (str): Filepath for the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
