import sys
import re
import string
import pandas as pd
import joblib
import sqlite3
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV ,RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk
from sklearn.pipeline import Pipeline

# Download NLTK resources
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
    - database_filepath (str): Path to the SQLite database file.
    
    Returns:
    - X (Series): Messages.
    - Y (DataFrame): Categories.
    - category_names (list): Names of category columns.
    """
    with sqlite3.connect(database_filepath) as conn:
        df = pd.read_sql_query("SELECT * FROM DisasterResponse", conn)

    X = df['message']
    Y = df.iloc[:, 4:]  # Assuming first 4 columns are: id, message, original, genre
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and preprocess text.

    Args:
        text (str): Input text.

    Returns:
        clean_tokens (list): List of cleaned and lemmatized tokens.
    """
    url_regex = r"(?:(?:https?|ftp):\/\/)?[\w\/\-?=%.]+\.[\w\/\-&?=%.]+"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token, tag in pos_tag(tokens):
        pos = 'n' if tag.startswith(
            "NN") else 'v' if tag.startswith('VB') else 'a'
        clean_token = lemmatizer.lemmatize(token, pos=pos).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with SMOTE and GridSearchCV.

    Returns:
        model (GridSearchCV): Optimized model with hyperparameter tuning.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(class_weight='balanced')
        ))
    ])

    # Hyperparameter tuning
    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 1)],
        'clf__estimator__n_estimators': [10],
        'clf__estimator__max_depth': [None],
        'clf__estimator__min_samples_split': [2],
    }

    model = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=3, cv=2, verbose=3, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance on the test set.

    Args:
        model: Trained model.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): Test targets.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)

    # Print classification report for each category
    for i, category in enumerate(category_names):
        print(f"\nEvaluating category: {category}")
        print(classification_report(
            Y_test.iloc[:, i], Y_pred[:, i], zero_division=0))

    # Calculate and print macro F1-score
    macro_f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    print(f"\nOverall Macro F1-Score: {macro_f1:.4f}")


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Args:
        model: Trained model.
        model_filepath (str): Path to save the model.
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print(f"Best Parameters: {model.best_params_}")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model to {model_filepath}...')
        save_model(model, model_filepath)

        print('Model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.\n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
