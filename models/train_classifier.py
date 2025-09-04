"""Disaster Response Classification Pipeline.

This module contains a complete machine learning pipeline for classifying disaster
response messages into multiple categories. The pipeline includes data loading,
text preprocessing, model training with hyperparameter tuning, evaluation, and
model persistence.

Typical usage example:
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
"""

import sys
import warnings
import sqlite3
import re
import pickle
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, f1_score
from sklearn.pipeline import Pipeline

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure required NLTK data is available


def ensure_nltk_data():
    """Download NLTK data if not already present."""
    import os
    nltk_data_dir = os.path.expanduser('~/nltk_data')

    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]

    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(
                    f"Warning: Could not download NLTK data '{download_name}': {e}")


# Call once at module level
ensure_nltk_data()

# Suppress sklearn UserWarnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Precompute cleaned stop words globally to avoid pickling issues


def _preprocess_stop_words():
    """Helper function to preprocess stop words - called once at import time."""
    lemmatizer = WordNetLemmatizer()
    processed = set()

    for word in ENGLISH_STOP_WORDS:
        tokens = word_tokenize(word.lower())
        for token in tokens:
            processed.add(lemmatizer.lemmatize(token.strip()))

    return list(processed)


# Global variable - computed once to avoid serialization issues with multiprocessing
CLEANED_STOP_WORDS = _preprocess_stop_words()


def load_data(database_filepath):
    """Load disaster response data from SQLite database.

    Connects to the specified SQLite database and loads the DisasterResponse table.
    Separates features (messages) from target variables (categories) and performs
    basic data cleaning by removing empty columns and filling missing values.

    Args:
        database_filepath (str): Filepath of the SQLite database containing the
                               DisasterResponse table.

    Returns:
        tuple: A tuple containing:
            - X (pd.Series): Messages column containing the text data to classify.
            - Y (pd.DataFrame): Target categories as binary indicators (0/1).
            - category_names (list): Names of target categories for reference.

    Raises:
        sqlite3.Error: If database connection or query fails.
        KeyError: If required columns are missing from the database table.
    """
    # Establish database connection using context manager for automatic cleanup
    with sqlite3.connect(database_filepath) as conn:
        # Load the entire DisasterResponse table into a DataFrame
        df = pd.read_sql_query("SELECT * FROM DisasterResponse", conn)

    # Separate features (message text) from other columns
    X = df['message']

    # Remove non-target columns to isolate category labels
    Y = df.drop(columns=['id', 'message', 'original',
                'genre'], errors='ignore')

    # Clean target data by removing completely empty columns
    Y = Y.dropna(axis=1, how='all')  # Remove empty target columns

    # Fill remaining NaN values with 0 (assuming binary classification)
    Y = Y.fillna(0)  # Fill remaining NaNs

    return X, Y, Y.columns.tolist()


def tokenize(text):
    """Tokenize and preprocess text for machine learning model input.

    This function performs comprehensive text preprocessing including normalization,
    tokenization, lemmatization, and stop word removal. The preprocessing steps
    must match exactly what was used during model training to ensure consistency.

    Args:
        text (str): Raw input text to be tokenized and processed.

    Returns:
        list: List of processed tokens (strings) ready for model input.

    Example:
        >>> tokenize("Help! We need water and medical supplies!")
        ['help', 'need', 'water', 'medical', 'supply']

    Note:
        This function is designed to be used as a custom tokenizer in 
        sklearn's TfidfVectorizer for consistent text preprocessing.
    """
    # Normalize text by removing non-alphanumeric characters and converting to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the normalized text into individual words
    tokens = word_tokenize(text)

    # Initialize lemmatizer for word root extraction
    lemmatizer = WordNetLemmatizer()

    # Remove stop words and lemmatize remaining tokens
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token).lower().strip()
              for token in tokens if token not in stop_words]

    return tokens


def build_model():
    """Build a machine learning pipeline with GridSearchCV for hyperparameter tuning.

    Creates a complete ML pipeline consisting of:
    1. TfidfVectorizer with custom tokenization for text feature extraction
    2. MultiOutputClassifier with RandomForestClassifier for multi-label classification
    3. GridSearchCV for automated hyperparameter optimization

    The pipeline is designed for multi-output classification where each message
    can belong to multiple disaster response categories simultaneously.

    Returns:
        GridSearchCV: Model selection object wrapping a pipeline with
                      TfidfVectorizer and MultiOutput RandomForestClassifier.
                      Configured for 3-fold cross-validation with weighted F1 scoring.

    Note:
        - Uses 'balanced' class weights to handle potential class imbalance
        - Disabled parallel processing (n_jobs=1) to avoid pickling issues
        - Random state set to 42 for reproducible results
    """
    # Build the machine learning pipeline using precomputed stop words
    pipeline = Pipeline([
        # Text feature extraction with custom tokenization
        ('vect', TfidfVectorizer(tokenizer=tokenize,
                                 stop_words=CLEANED_STOP_WORDS,
                                 token_pattern=None)),  # Use custom tokenizer
        # Multi-output classifier for handling multiple categories per message
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            class_weight='balanced',  # Handle class imbalance
            random_state=42)))        # Ensure reproducible results
    ])

    # Define hyperparameter grid for optimization
    parameters = {
        'clf__estimator__n_estimators': [50, 100],           # Number of trees
        # Min samples to split node
        'clf__estimator__min_samples_split': [2, 4],
        # Maximum tree depth
        'clf__estimator__max_depth': [None, 10, 20],
        # Features per split
        'clf__estimator__max_features': ['sqrt', 'log2']
    }

    # Return GridSearchCV object with specified parameters
    # Note: Using n_jobs=1 to avoid pickling issues with custom tokenizer
    return GridSearchCV(pipeline, param_grid=parameters,
                        scoring='f1_weighted',  # Use weighted F1 for imbalanced data
                        verbose=2,              # Show progress during search
                        n_jobs=1,               # Disable multiprocessing to avoid pickling issues
                        cv=3)                   # 3-fold cross-validation


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate trained model performance and generate comprehensive reports.

    Performs detailed evaluation of the multi-output classification model including:
    - Individual category classification reports
    - Confusion matrices for each category (saved as PNG files)
    - Overall performance metrics
    - CSV export of all metrics for further analysis

    Args:
        model (GridSearchCV): Trained model object with best parameters.
        X_test (pd.Series): Test features (message texts).
        Y_test (pd.DataFrame): Test targets (binary category indicators).
        category_names (list): List of category names corresponding to Y_test columns.

    Side Effects:
        - Prints detailed classification reports to console
        - Saves confusion matrix plots as PNG files for each category
        - Saves classification metrics to 'classification_report.csv'
        - Prints overall weighted F1 score

    Note:
        Uses zero_division=0 to handle cases where precision/recall is undefined
        due to no predicted positive cases for a category.
    """
    # Generate predictions for the test set
    Y_pred = model.predict(X_test)
    reports = []

    # Evaluate each category individually
    for i, col in enumerate(category_names):
        print(f"\nCategory: {col}")

        # Generate and display detailed classification report
        report = classification_report(
            Y_test[col], Y_pred[:, i], zero_division=0)
        print(report)

        # Calculate individual metrics for CSV export
        precision, recall, f1, _ = precision_recall_fscore_support(
            Y_test[col], Y_pred[:, i], average='binary', zero_division=0
        )

        # Store metrics for comprehensive reporting
        reports.append({
            'category': col,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        # Generate and save confusion matrix visualization
        cm = confusion_matrix(Y_test[col], Y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix: {col}')
        plt.savefig(f'confusion_matrix_{col}.png')
        plt.close()  # Close plot to free memory

    # Export detailed metrics to CSV for further analysis
    pd.DataFrame(reports).to_csv("classification_report.csv", index=False)

    # Calculate and display overall model performance
    overall_f1 = f1_score(Y_test.values, Y_pred,
                          average='weighted', zero_division=0)
    print(f"\nOverall Weighted F1 Score: {overall_f1:.4f}")


def save_model(model, model_filepath):
    """Save trained model to a pickle file for later use.

    Serializes the complete trained model (including the best parameters found
    by GridSearchCV) to a pickle file for deployment or future predictions.

    Args:
        model (GridSearchCV): Trained model object containing the best estimator
                             and all fitted parameters.
        model_filepath (str): Destination filepath where the pickle file will be saved.
                             Should include .pkl extension.

    Raises:
        IOError: If the file cannot be written to the specified path.
        PickleError: If the model object cannot be serialized.

    Note:
        The saved model includes both the pipeline and the best hyperparameters
        found during grid search, making it ready for immediate deployment.
    """
    # Save model using pickle with binary write mode
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """Main function to execute the complete ML pipeline.

    Orchestrates the entire machine learning workflow from data loading through
    model training, evaluation, and persistence. Includes command-line argument
    validation and error handling for robust execution.

    Command Line Args:
        database_filepath (str): Path to SQLite database containing disaster response data.
        model_filepath (str): Path where the trained model pickle file should be saved.

    Steps:
        1. Load and validate command line arguments
        2. Load data from the specified SQLite database
        3. Split data into training and test sets (80/20 split)
        4. Build ML pipeline with hyperparameter tuning
        5. Train model using grid search cross-validation
        6. Evaluate model performance on test set
        7. Save trained model to specified pickle file

    Example:
        python train_classifier.py ../data/DisasterResponse.db classifier.pkl

    Raises:
        SystemExit: If incorrect number of command line arguments provided.
    """
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("""Please provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to save the model to as the second argument.

Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl""")
        sys.exit(1)

    # Extract command line arguments
    database_filepath, model_filepath = sys.argv[1], sys.argv[2]

    # Step 1: Load data from database
    print(f'\nLoading data...\n    DATABASE: {database_filepath}')
    X, Y, category_names = load_data(database_filepath)

    # Step 2: Split data into training and test sets with stratified sampling
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # Step 3: Build ML pipeline with hyperparameter tuning
    print('\nBuilding model...')
    model = build_model()

    # Step 4: Train model using grid search cross-validation
    print('\nTraining model...')
    model.fit(X_train, Y_train)

    # Step 5: Evaluate model performance on held-out test set
    print('\nEvaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    # Step 6: Save trained model for deployment
    print(f'\nSaving model...\n    MODEL: {model_filepath}')
    save_model(model, model_filepath)

    print('\nTrained model saved!')


# Execute main function when script is run directly
if __name__ == '__main__':
    main()
