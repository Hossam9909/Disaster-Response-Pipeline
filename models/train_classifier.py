import os
import sys
import re
import string
import pandas as pd
import joblib  # Replaced pickle with joblib
from sqlalchemy import create_engine
from imblearn.pipeline import Pipeline  # Changed to imblearn's Pipeline
from imblearn.over_sampling import SMOTE
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# ================== NEW: Added SMOTE and class weights ==================


def build_model():
    """Builds a machine learning pipeline with SMOTE and improved GridSearchCV."""
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('smote', SMOTE(random_state=42)),  # Added SMOTE
        ('clf', MultiOutputClassifier(
            # Added class weights
            RandomForestClassifier(class_weight='balanced')
        ))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [100, 200],  # Expanded options
        'clf__estimator__max_depth': [None, 10, 20],
        'clf__estimator__min_samples_split': [2, 5],  # More flexible
        'smote__k_neighbors': [3, 5]  # SMOTE hyperparameters
    }

    return GridSearchCV(pipeline, parameters, cv=3, verbose=3, n_jobs=-1)

# ================== NEW: Enhanced evaluation metrics ==================


def evaluate_model(model, x_test, y_test, category_names):
    """Improved evaluation with macro F1-score and per-category metrics."""
    y_pred = model.predict(x_test)

    # Overall metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\nMacro F1-Score: {macro_f1:.4f}")

    # Per-category report
    for category in category_names:
        f1 = f1_score(y_test[category], y_pred[:,
                      y_test.columns.get_loc(category)], zero_division=0)
        print(f"{category:<30} F1: {f1:.4f}")

    # Full classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=category_names, zero_division=0))

# ================== UPDATED: Replaced pickle with joblib ==================


def save_model(model, model_filepath):
    """Saves model using joblib for efficiency."""
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    joblib.dump(model, model_filepath)
