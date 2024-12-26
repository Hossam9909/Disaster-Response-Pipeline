import sys
import re
import string
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Extracts whether the starting word in text is a verb.
    """

    def starting_verb(self, text):
        """Checks if the first word of a sentence is a verb or 'RT'.

        Args:
            text (str): Input text.

        Returns:
            bool: True if the first word is a verb or 'RT', False otherwise.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            tokens = tokenize(sentence)
            if not tokens:  # Handle empty sentences
                continue
            pos_tags = nltk.pos_tag([tokens[0]])  # pos tag only the first word
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """Fit method (does nothing in this case).

        Args:
            X: Input data.
            y: Target data (optional).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """Transforms input data to a Pandas DataFrame of starting verb features.

        Args:
            X (Series or array-like): Input text data.

        Returns:
            DataFrame: Pandas DataFrame with binary starting verb features.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """Builds a machine learning pipeline with GridSearchCV for text classification.

    The pipeline includes text processing (tokenization, TF-IDF),
    a custom feature extractor (StartingVerbExtractor), and a
    RandomForestClassifier. GridSearchCV is used for hyperparameter tuning.

    Returns:
        GridSearchCV: A GridSearchCV object fitted with the pipeline and parameters.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_depth': [None, 10, 20],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)

    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """Evaluates model performance on the test data using classification metrics.

    Prints a classification report for each category.

    Args:
        model: Trained machine learning model.
        x_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test targets.
        category_names (list): List of category names.

    Returns:
        None
    """
    y_pred = model.predict(x_test)

    # Convert predictions to DataFrame for easier comparison and reporting
    y_pred_df = pd.DataFrame(
        y_pred, columns=category_names, index=y_test.index)

    for category in category_names:
        print(f"Category: {category}\n")
        print(classification_report(y_test[category], y_pred_df[category]))

    # Calculate and print overall accuracy
    overall_accuracy = (y_pred == y_test.values).mean()
    print(f"Overall Accuracy: {overall_accuracy}")


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
