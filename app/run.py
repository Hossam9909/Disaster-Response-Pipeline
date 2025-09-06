"""
Disaster Response Classification Web Application.

This Flask application provides a web interface for classifying disaster-related messages
using machine learning models, visualizing data patterns, and providing emergency response
recommendations. The application includes features for message search, word analysis,
and interactive data visualization.

Author: Disaster Response Team
Version: 1.0
"""

import os
import re
import json
import pandas as pd
import joblib
import plotly
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from sqlalchemy import create_engine
from flask import Flask, render_template, request, redirect, flash, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from functools import lru_cache
from collections import Counter
import nltk
from nltk.corpus import stopwords


def ensure_nltk_data():
    """Download NLTK data if not already present."""
    nltk_data_dir = os.path.expanduser('~/nltk_data')

    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]

    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK data '{download_name}': {e}")


# Call
ensure_nltk_data()


# ================== TOKENIZATION FUNCTION ==================
# This function needs to be defined before loading the model
# It should match exactly what was used during model training


def tokenize(text):
    """
    Tokenize and preprocess text for machine learning model input.

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
    """
    # Handle None/NaN values
    if pd.isna(text) or text is None:
        return []

    # Convert to string to handle numeric inputs
    text = str(text)

    # Normalize text by removing non-alphanumeric characters and converting to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the normalized text into individual words
    tokens = word_tokenize(text)

    # Initialize lemmatizer for word root extraction
    lemmatizer = WordNetLemmatizer()

    # Remove stop words and lemmatize remaining tokens
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token).lower().strip()
              for token in tokens if token not in stop_words and len(token) > 2]

    return tokens


# Initialize Flask application and database configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
# Generate random secret key for session management
app.secret_key = os.urandom(24)
db = SQLAlchemy(app)


# ================== Feedback Model ==================


class Feedback(db.Model):
    """
    SQLAlchemy model for storing user feedback on message classifications.

    This model tracks user feedback on the accuracy of machine learning predictions
    to enable continuous improvement of the classification system.

    Attributes:
        id (int): Primary key for the feedback record.
        message (str): The original message that was classified (max 500 characters).
        prediction (str): The model's prediction result (max 500 characters).
        is_correct (bool): Whether the user marked the prediction as accurate.
    """

    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(500))
    prediction = db.Column(db.String(500))
    is_correct = db.Column(db.Boolean)


# ================== Recommendations ==================


RECOMMENDATIONS = {
    'water': {
        'text': 'Contact water assistance organizations',
        'links': [
            {'name': 'Water.org', 'url': 'https://water.org'},
            {'name': 'UNICEF Water', 'url': 'https://www.unicef.org/water-sanitation-hygiene'}
        ]
    },
    'medical_help': {
        'text': 'Alert medical support organizations and contact local emergency services',
        'links': [
            {'name': 'Red Cross', 'url': 'https://www.redcross.org'},
            {'name': 'Doctors Without Borders', 'url': 'https://www.doctorswithoutborders.org'}
        ]
    },
    'food': {
        'text': 'Reach out to food assistance organizations',
        'links': [
            {'name': 'World Food Programme', 'url': 'https://www.wfp.org'},
            {'name': 'Action Against Hunger', 'url': 'https://www.actionagainsthunger.org'}
        ]
    },
    'shelter': {
        'text': 'Contact emergency shelter organizations and local emergency services',
        'links': [
            {'name': 'UN Habitat', 'url': 'https://unhabitat.org'},
            {'name': 'Habitat for Humanity', 'url': 'https://www.habitat.org'}
        ]
    },
    'refugees': {
        'text': 'Contact refugee assistance programs',
        'links': [
            {'name': 'UNHCR', 'url': 'https://www.unhcr.org'},
            {'name': 'International Rescue Committee', 'url': 'https://www.rescue.org'}
        ]
    },
    'direct_report': {
        'text': 'Forward to appropriate emergency response authorities and contact local emergency services',
        'links': [
            {'name': 'UN OCHA', 'url': 'https://www.unocha.org'}
        ]
    }
}


# ================== Load Model and Data ==================

# Construct path to disaster response database
db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DisasterResponse.db')

try:
    # Create database engine and load disaster response data
    engine = create_engine(f'sqlite:///{os.path.abspath(db_path)}')
    df = pd.read_sql_table('DisasterResponse', engine)
    print(f"✅ Successfully loaded {len(df)} records from database")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame shape: {df.shape}")

    # Debug output: Check first few messages to verify data integrity
    if 'message' in df.columns:
        print("First 3 messages in original data:")
        for i, msg in enumerate(df['message'].head(3)):
            print(f"  Row {i}: '{str(msg)[:100]}...'")
    else:
        print("⚠️ 'message' column not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")

except Exception as e:
    print(f"❌ Error loading database: {e}")
    # Create empty dataframe as fallback to prevent application crashes
    df = pd.DataFrame(columns=['message', 'genre'])


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained machine learning model with comprehensive error handling.

    This function attempts to load a pre-trained classifier model from the models
    directory. It includes multiple fallback loading strategies to handle different
    types of serialization issues that may occur.

    Returns:
        object or None: The loaded machine learning model if successful, None if failed.

    Note:
        The model file should be located at '../models/classifier.pkl' relative to
        this script. The function uses LRU cache to avoid reloading the model
        multiple times during application runtime.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl')

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Model loaded successfully from: {os.path.abspath(model_path)}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file not found at: {os.path.abspath(model_path)}")
        print("Make sure classifier.pkl is in the models/ directory")
        return None


# Load the machine learning model at application startup
model = load_model()


# ================== Helper Functions ==================


def truncate_text(text, max_length=100):
    """
    Truncate text to specified length with ellipsis for display purposes.

    Args:
        text (str): The text to be truncated.
        max_length (int, optional): Maximum length of the returned text. Defaults to 100.

    Returns:
        str: Truncated text with ellipsis if truncation occurred, original text otherwise.

    Example:
        >>> truncate_text("This is a very long message that needs truncation", 20)
        'This is a very long ...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def search_messages(df, query, limit=50):
    """
    Search for messages containing a specific query string.

    Performs case-insensitive substring search within the message column
    of the provided DataFrame and returns matching results.

    Args:
        df (pd.DataFrame): DataFrame containing messages to search.
        query (str): Search term to look for in messages.
        limit (int, optional): Maximum number of results to return. Defaults to 50.

    Returns:
        pd.DataFrame: Filtered DataFrame containing messages that match the query.

    Note:
        If query is empty or whitespace-only, returns the first 'limit' rows.
        Search is performed on the 'message' column only.
    """
    if not query.strip():
        return df.head(limit)

    # Perform case-insensitive search in message column
    mask = df['message'].str.contains(query, case=False, na=False)
    filtered_df = df[mask]

    # Apply result limit if specified
    if limit > 0:
        filtered_df = filtered_df.head(limit)

    return filtered_df


def analyze_word_in_dataset(word, df):
    """
    Perform comprehensive analysis of a specific word's usage patterns in the dataset.

    This function analyzes how a specific word appears across the disaster response
    dataset, including frequency analysis, category associations, genre distribution,
    and co-occurring word patterns.

    Args:
        word (str): The word to analyze within the dataset.
        df (pd.DataFrame): The disaster response dataset to analyze.

    Returns:
        dict or None: Dictionary containing detailed analysis results, or None if 
                     dataset is empty or missing required columns.

    The returned dictionary contains:
        - word (str): The analyzed word
        - total_occurrences (int): Total times word appears in all messages
        - message_count (int): Number of messages containing the word
        - percentage (float): Percentage of total messages containing the word
        - category_distribution (dict): Breakdown by disaster categories
        - genre_distribution (dict): Breakdown by message source/genre
        - co_occurring_words (list): Words frequently appearing with target word
        - sample_messages (list): Example messages containing the word
    """
    if df.empty or 'message' not in df.columns:
        return None

    # Clean and normalize the search word
    word_clean = word.lower().strip()

    # Find all messages containing the target word
    mask = df['message'].str.contains(word_clean, case=False, na=False)
    word_messages = df[mask]

    # Handle case where no messages contain the word
    if len(word_messages) == 0:
        return {
            'word': word,
            'total_occurrences': 0,
            'message_count': 0,
            'percentage': 0,
            'category_distribution': {},
            'genre_distribution': {},
            'co_occurring_words': [],
            'sample_messages': []
        }

    # Identify category columns (exclude metadata columns)
    category_cols = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]

    # Calculate category distribution for messages containing the word
    category_distribution = {}
    for cat in category_cols:
        if cat in word_messages.columns:
            count = word_messages[cat].sum()
            percentage = (count / len(word_messages)) * 100
            category_distribution[cat.replace('_', ' ').title()] = {
                'count': int(count),
                'percentage': round(percentage, 1)
            }

    # Calculate genre distribution for messages containing the word
    genre_distribution = {}
    if 'genre' in word_messages.columns:
        genre_counts = word_messages['genre'].value_counts()
        for genre, count in genre_counts.items():
            percentage = (count / len(word_messages)) * 100
            genre_distribution[genre] = {
                'count': int(count),
                'percentage': round(percentage, 1)
            }

    # Find words that frequently co-occur with the target word
    stop_words = set(stopwords.words('english'))
    stop_words.update(['said', 'say', 'get', 'go', 'know', 'make', 'come', 'take', 'see', 'want'])
    stop_words.add(word_clean)  # Remove the searched word itself

    # Combine all messages containing the word and analyze co-occurring words
    word_messages_text = ' '.join(word_messages['message'].dropna().astype(str)).lower()
    tokens = word_tokenize(word_messages_text)
    words = [w for w in tokens if w.isalpha() and len(w) > 2 and w not in stop_words]
    co_occurring_words = Counter(words).most_common(10)

    # Count total occurrences of the word across all messages
    total_occurrences = sum(msg.lower().count(word_clean) for msg in df['message'].dropna().astype(str))

    # Get sample messages for qualitative analysis
    sample_messages = word_messages['message'].head(5).tolist()

    return {
        'word': word,
        'total_occurrences': total_occurrences,
        'message_count': len(word_messages),
        'percentage': round((len(word_messages) / len(df)) * 100, 2),
        'category_distribution': category_distribution,
        'genre_distribution': genre_distribution,
        'co_occurring_words': co_occurring_words,
        'sample_messages': sample_messages
    }


def create_word_visualizations(word_analysis):
    """
    Create Plotly visualizations for word analysis results.

    Generates interactive charts to visualize patterns and distributions
    related to a specific word's usage in the disaster response dataset.

    Args:
        word_analysis (dict): Results from analyze_word_in_dataset function.

    Returns:
        list: List of JSON-encoded Plotly figure objects ready for web rendering.
              Returns empty list if analysis data is invalid or visualization fails.

    Generated visualizations include:
        1. Category distribution bar chart
        2. Genre distribution pie chart  
        3. Co-occurring words frequency chart
        4. Usage statistics summary chart
    """
    if not word_analysis or word_analysis['message_count'] == 0:
        return []

    visuals = []
    word = word_analysis['word']

    try:
        # 1. Category Distribution for messages containing this word
        if word_analysis['category_distribution']:
            categories = list(word_analysis['category_distribution'].keys())
            percentages = [data['percentage'] for data in word_analysis['category_distribution'].values()]
            counts = [data['count'] for data in word_analysis['category_distribution'].values()]

            # Only show categories with non-zero values to avoid clutter
            non_zero_data = [(cat, perc, count) for cat, perc, count in zip(categories, percentages, counts) if count > 0]

            if non_zero_data:
                categories, percentages, counts = zip(*non_zero_data)

                fig1 = px.bar(
                    x=list(categories),
                    y=list(percentages),
                    labels={'x': 'Category', 'y': 'Percentage of Messages'},
                    title=f'Categories Associated with "{word.title()}" Messages',
                    text=list(counts),
                    color=list(percentages),
                    color_continuous_scale='viridis'
                )
                fig1.update_traces(texttemplate='%{text}', textposition='outside')
                fig1.update_layout(xaxis_tickangle=-45, height=500)
                visuals.append(json.dumps(fig1, cls=PlotlyJSONEncoder))

        # 2. Genre Distribution pie chart for messages containing this word
        if word_analysis['genre_distribution']:
            genres = list(word_analysis['genre_distribution'].keys())
            genre_counts = [data['count'] for data in word_analysis['genre_distribution'].values()]

            fig2 = px.pie(
                values=genre_counts,
                names=genres,
                title=f'Message Sources for "{word.title()}" Messages'
            )
            visuals.append(json.dumps(fig2, cls=PlotlyJSONEncoder))

        # 3. Co-occurring Words frequency chart
        if word_analysis['co_occurring_words']:
            co_words, co_counts = zip(*word_analysis['co_occurring_words'])

            fig3 = px.bar(
                x=list(co_counts),
                y=list(co_words),
                orientation='h',
                title=f'Words Frequently Appearing with "{word.title()}"',
                labels={'x': 'Frequency', 'y': 'Words'},
                color=list(co_counts),
                color_continuous_scale='plasma'
            )
            fig3.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            visuals.append(json.dumps(fig3, cls=PlotlyJSONEncoder))

        # 4. Word Usage Statistics summary chart
        stats_data = {
            'Metric': ['Total Messages', 'Messages with Word', 'Word Occurrences', 'Coverage %'],
            'Value': [
                len(df),
                word_analysis['message_count'],
                word_analysis['total_occurrences'],
                word_analysis['percentage']
            ]
        }

        fig4 = px.bar(
            x=stats_data['Metric'],
            y=stats_data['Value'],
            title=f'Usage Statistics for "{word.title()}"',
            text=stats_data['Value'],
            color=stats_data['Value'],
            color_continuous_scale='blues'
        )
        fig4.update_traces(texttemplate='%{text}', textposition='outside')
        fig4.update_layout(height=400)
        visuals.append(json.dumps(fig4, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error creating word visualizations: {e}")

    return visuals


def create_advanced_word_visualizations(word, df, word_messages):
    """
    Create advanced statistical visualizations for comprehensive word analysis.

    Generates sophisticated analytical charts that provide deeper insights into
    how a specific word relates to various aspects of the disaster response dataset,
    including category relationships, message complexity, and comparative analysis.

    Args:
        word (str): The word being analyzed.
        df (pd.DataFrame): Complete disaster response dataset.
        word_messages (pd.DataFrame): Subset of messages containing the target word.

    Returns:
        list: List of JSON-encoded Plotly figure objects for advanced visualizations.
              Returns empty list if visualization creation fails.

    Generated advanced visualizations include:
        1. Category comparison (word vs overall dataset)
        2. Multi-category message analysis
        3. Category co-occurrence heatmap
        4. Message length comparison analysis
        5. Language distribution analysis
        6. Top category combination patterns
        7. Natural disaster type associations
    """
    visuals = []
    # Identify category columns by excluding known metadata columns
    category_cols = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]

    try:
        # 1. Category Comparison: Messages with word vs Overall Dataset
        overall_category_means = df[category_cols].mean() * 100
        word_category_means = word_messages[category_cols].mean() * 100

        comparison_data = pd.DataFrame({
            'Category': [col.replace('_', ' ').title() for col in category_cols],
            'Overall Dataset %': overall_category_means.values,
            'Messages with Word %': word_category_means.values
        })

        # Only show categories where there's a significant difference (>5%)
        comparison_data['Difference'] = comparison_data['Messages with Word %'] - comparison_data['Overall Dataset %']
        significant_diff = comparison_data[abs(comparison_data['Difference']) > 5].head(15)

        if not significant_diff.empty:
            fig1 = px.bar(
                significant_diff,
                x='Category',
                y=['Overall Dataset %', 'Messages with Word %'],
                title=f'Category Distribution: "{word.title()}" vs Overall Dataset',
                barmode='group',
                labels={'value': 'Percentage (%)', 'variable': 'Dataset Type'},
                height=500
            )
            fig1.update_layout(xaxis_tickangle=-45)
            visuals.append(json.dumps(fig1, cls=PlotlyJSONEncoder))

        # 2. Multi-category Messages Analysis - shows message complexity
        word_messages_copy = word_messages.copy()
        word_messages_copy['category_count'] = word_messages_copy[category_cols].sum(axis=1)

        category_count_dist = word_messages_copy['category_count'].value_counts().sort_index()

        fig2 = px.bar(
            x=category_count_dist.index,
            y=category_count_dist.values,
            title=f'Number of Categories per Message containing "{word.title()}"',
            labels={'x': 'Number of Categories', 'y': 'Number of Messages'},
            text=category_count_dist.values
        )
        fig2.update_traces(textposition='outside')
        visuals.append(json.dumps(fig2, cls=PlotlyJSONEncoder))

        # 3. Category Co-occurrence Network (Correlation Heatmap)
        word_categories = word_messages[category_cols]
        # Only include categories that appear in word messages
        active_categories = word_categories.columns[word_categories.sum() > 0]

        if len(active_categories) > 1:
            correlation_matrix = word_messages[active_categories].corr()

            fig3 = px.imshow(
                correlation_matrix,
                title=f'Category Relationships in "{word.title()}" Messages',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto=True
            )
            fig3.update_layout(height=600)
            visuals.append(json.dumps(fig3, cls=PlotlyJSONEncoder))

        # 4. Message Length Analysis - comparing average lengths
        word_messages_copy['message_length'] = word_messages_copy['message'].str.len()
        overall_avg_length = df['message'].str.len().mean()
        word_avg_length = word_messages_copy['message_length'].mean()

        length_comparison = {
            'Dataset': ['Overall Dataset', f'Messages with "{word.title()}"'],
            'Average Length': [overall_avg_length, word_avg_length]
        }

        fig4 = px.bar(
            x=length_comparison['Dataset'],
            y=length_comparison['Average Length'],
            title=f'Average Message Length Comparison',
            text=length_comparison['Average Length'],
            color=length_comparison['Average Length'],
            color_continuous_scale='viridis'
        )
        fig4.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        visuals.append(json.dumps(fig4, cls=PlotlyJSONEncoder))

        # 5. Language Detection Analysis (Original vs Translated Messages)
        if 'original' in word_messages.columns:
            has_translation = word_messages['original'].notna() & (word_messages['original'] != word_messages['message'])
            translation_stats = {
                'Type': ['English Only', 'Has Translation'],
                'Count': [
                    len(word_messages) - has_translation.sum(),
                    has_translation.sum()
                ]
            }

            fig5 = px.pie(
                values=translation_stats['Count'],
                names=translation_stats['Type'],
                title=f'Language Distribution for "{word.title()}" Messages'
            )
            visuals.append(json.dumps(fig5, cls=PlotlyJSONEncoder))

        # 6. Top Category Combinations Analysis
        # Find most common combinations of categories
        category_combinations = []
        for _, row in word_messages.iterrows():
            active_cats = [col for col in category_cols if row[col] == 1]
            if active_cats:
                # Limit to 3 categories for readability
                category_combinations.append(' + '.join(sorted(active_cats)[:3]))

        combo_counts = Counter(category_combinations).most_common(10)
        if combo_counts:
            combo_names, combo_vals = zip(*combo_counts)

            fig6 = px.bar(
                x=list(combo_vals),
                y=[name.replace('_', ' ').title() for name in combo_names],
                orientation='h',
                title=f'Most Common Category Combinations for "{word.title()}"',
                labels={'x': 'Frequency', 'y': 'Category Combinations'},
                color=list(combo_vals),
                color_continuous_scale='plasma'
            )
            fig6.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            visuals.append(json.dumps(fig6, cls=PlotlyJSONEncoder))

        # 7. Natural Disaster Type Distribution Analysis
        disaster_categories = [col for col in category_cols if any(keyword in col.lower() for keyword in
                              ['weather', 'flood', 'storm', 'fire', 'earthquake', 'cold'])]

        if disaster_categories:
            disaster_counts = word_messages[disaster_categories].sum()
            disaster_counts = disaster_counts[disaster_counts > 0].sort_values(ascending=False)

            if not disaster_counts.empty:
                fig7 = px.bar(
                    x=[col.replace('_', ' ').title() for col in disaster_counts.index],
                    y=disaster_counts.values,
                    title=f'Natural Disaster Types Associated with "{word.title()}"',
                    labels={'x': 'Disaster Type', 'y': 'Number of Messages'},
                    color=disaster_counts.values,
                    color_continuous_scale='reds'
                )
                fig7.update_layout(xaxis_tickangle=-45)
                visuals.append(json.dumps(fig7, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error creating advanced visualizations: {e}")

    return visuals


def get_word_insights(word, df, word_messages):
    """
    Generate textual insights about word usage patterns and characteristics.

    Analyzes the relationship between a specific word and various dataset characteristics
    to provide human-readable insights about usage patterns, associations, and anomalies.

    Args:
        word (str): The word being analyzed.
        df (pd.DataFrame): Complete disaster response dataset for comparison.
        word_messages (pd.DataFrame): Messages containing the target word.

    Returns:
        list: List of insight strings describing notable patterns and characteristics.
              Each insight is a complete sentence describing a significant finding.

    Generated insights may include:
        - Primary category associations
        - Genre preferences
        - Message complexity comparisons
        - Usage frequency patterns
    """
    # Identify category columns for analysis
    category_cols = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]
    insights = []

    # Analyze strongest category association
    word_category_means = word_messages[category_cols].mean()
    top_category = word_category_means.idxmax()
    top_percentage = word_category_means.max() * 100

    if top_percentage > 10:
        insights.append(f'Most strongly associated with {top_category.replace("_", " ").title()} ({top_percentage:.1f}% of messages)')

    # Analyze genre preferences
    if 'genre' in word_messages.columns:
        top_genre = word_messages['genre'].mode()[0] if not word_messages['genre'].mode().empty else 'Unknown'
        genre_percentage = (word_messages['genre'] == top_genre).mean() * 100
        insights.append(f'Most common in {top_genre} messages ({genre_percentage:.1f}%)')

    # Analyze message complexity compared to overall dataset
    avg_categories = word_messages[category_cols].sum(axis=1).mean()
    overall_avg = df[category_cols].sum(axis=1).mean()

    if avg_categories > overall_avg * 1.2:
        insights.append(f'Messages tend to be more complex (avg {avg_categories:.1f} categories vs {overall_avg:.1f} overall)')
    elif avg_categories < overall_avg * 0.8:
        insights.append(f'Messages tend to be simpler (avg {avg_categories:.1f} categories vs {overall_avg:.1f} overall)')

    return insights

# ================== Visualizations ==================


@lru_cache(maxsize=1)
def create_visualizations():
    """Create all visualizations for the dashboard.

    This function generates various data visualizations including category
    distribution, genre distribution, correlation heatmaps, message length
    distribution, and word frequency. Uses LRU cache to avoid regenerating
    visualizations on repeated calls.

    Returns:
        list: A list of JSON-encoded Plotly figures ready for rendering in
              templates. Returns empty list if no data is available or if
              errors occur.

    Note:
        Assumes global 'df' DataFrame is available with expected columns:
        - Numeric columns (indices 4+) for categories
        - 'genre' column for message genres  
        - 'message' column for message text
    """
    # Check if DataFrame is available and not empty
    if df.empty:
        print("⚠️ No data available for visualizations")
        return []

    visuals = []

    try:
        # 1. Category Distribution Bar Chart
        # Extract numeric columns (categories) starting from index 4
        category_df = df.iloc[:, 4:].select_dtypes(include=[np.number])

        if not category_df.empty:
            # Calculate sum for each category and sort in descending order
            category_counts = category_df.sum().sort_values(ascending=False)

            # Create interactive bar chart with color mapping
            fig1 = px.bar(
                x=category_counts.index.str.replace('_', ' ').str.title(),
                y=category_counts.values,
                labels={'x': 'Category', 'y': 'Count'},
                title='Message Category Distribution',
                color=category_counts.values,
                color_continuous_scale='viridis'
            )
            # Rotate x-axis labels for readability
            fig1.update_layout(xaxis_tickangle=-45, height=500)
            visuals.append(json.dumps(fig1, cls=PlotlyJSONEncoder))

        # 2. Genre Distribution Pie Chart
        if 'genre' in df.columns:
            # Count occurrences of each genre
            genre_counts = df['genre'].value_counts()

            # Create pie chart for genre distribution
            fig2 = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title='Message Distribution by Genre'
            )
            visuals.append(json.dumps(fig2, cls=PlotlyJSONEncoder))

        # 3. Category Correlation Heatmap (top 15 categories for readability)
        if not category_df.empty and len(category_df.columns) > 1:
            # Select top 15 categories to avoid cluttered heatmap
            top_categories = category_counts.head(15).index
            # Calculate correlation matrix
            corr_matrix = df[top_categories].corr()

            # Create correlation heatmap with diverging color scale
            fig3 = px.imshow(
                corr_matrix,
                title='Top 15 Categories Correlation Heatmap',
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig3.update_layout(height=600)
            visuals.append(json.dumps(fig3, cls=PlotlyJSONEncoder))

        # 4. Message Length Distribution
        if 'message' in df.columns:
            # Create copy to avoid modifying original DataFrame
            df_copy = df.copy()

            # Calculate message length in characters, handling NaN values
            df_copy['msg_len'] = df_copy['message'].apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            )

            # Create histogram for message length distribution
            fig4 = px.histogram(
                df_copy,
                x='msg_len',
                nbins=50,
                title='Distribution of Message Lengths',
                labels={
                    'msg_len': 'Message Length (characters)',
                    'count': 'Frequency'
                },
                color_discrete_sequence=['skyblue']
            )
            fig4.update_layout(height=400)
            visuals.append(json.dumps(fig4, cls=PlotlyJSONEncoder))

        # 5. Top Words Frequency
        if 'message' in df.columns:
            # Define stop words to exclude from frequency analysis
            stop_words = set(stopwords.words('english'))
            stop_words.update([
                'said', 'say', 'get', 'go', 'know',
                'make', 'come', 'take', 'see', 'want'
            ])

            # Combine all messages into single text and convert to lowercase
            all_words = ' '.join(df['message'].dropna().astype(str)).lower()
            tokens = word_tokenize(all_words)

            # Filter words: alphabetic, length > 2, not in stop words
            words = [
                word for word in tokens
                if word.isalpha() and len(word) > 2 and word not in stop_words
            ]

            # Get 15 most common words
            common_words = Counter(words).most_common(15)

            if common_words:
                # Create DataFrame for plotting
                word_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

                # Create horizontal bar chart for word frequency
                fig5 = px.bar(
                    word_df,
                    x='Count',
                    y='Word',
                    orientation='h',
                    title='Top 15 Most Frequent Words',
                    color='Count',
                    color_continuous_scale='plasma'
                )
                fig5.update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                visuals.append(json.dumps(fig5, cls=PlotlyJSONEncoder))
            else:
                # Fallback empty chart if no words found
                fig5 = go.Figure()
                fig5.update_layout(title='Top Words - No Data Available')
                visuals.append(json.dumps(fig5, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")

    return visuals


# ================== Routes ==================


@app.route('/')
def index():
    """Main page with visualizations and message table.

    Renders the main dashboard page containing:
    - Interactive data visualizations (charts, graphs, heatmaps)
    - Searchable and filterable message table
    - Category distribution analysis

    Query Parameters:
        limit (int, optional): Maximum number of messages to display.
                              Defaults to 50.
        search (str, optional): Search query to filter messages.
                               Defaults to empty string.

    Returns:
        str: Rendered HTML template with visualizations and table data.

    Template Variables:
        table_data (list): List of dictionaries containing message and
                          category data
        columns (list): Column headers for the data table
        category_cols (list): List of category column names
        row_limit (int): Current row limit setting
        search_query (str): Current search query
        graphs (list): JSON-encoded Plotly figures
        graph_titles (list): Titles for each graph
        ids (list): HTML element IDs for each graph
    """
    # Get query parameters with defaults
    limit = request.args.get('limit', default=50, type=int)
    search_query = request.args.get('search', default='', type=str)

    # Check if data is available
    if df.empty:
        flash("No data available. Please check database connection.")
        return render_template(
            'master.html',
            table_data=[],
            columns=[],
            graphs=[],
            graph_titles=[],
            ids=[]
        )

    # Get all category columns by excluding known non-category ones
    category_cols = [
        col for col in df.columns
        if col not in ['id', 'message', 'original', 'genre']
    ]

    # Search and filter data based on query parameters
    if search_query:
        df_filtered = search_messages(df, search_query, limit)
        if df_filtered.empty:
            flash(f"No messages found containing '{search_query}'")
            df_filtered = df.head(limit)
    else:
        df_filtered = df.head(limit) if limit > 0 else df

    # Prepare table data for rendering in the template
    table_data = []
    for _, row in df_filtered.iterrows():
        row_dict = {}

        # Add message content
        row_dict['message'] = str(row.get('message', 'No message available'))

        # Add genre information
        row_dict['genre'] = str(row.get('genre', '-'))

        # Add categories with consistent naming for the template
        for cat in category_cols:
            template_key = cat.lower().replace(' ', '_')
            if cat in row:
                # Convert to integer, handling NaN values
                row_dict[template_key] = (
                    int(row[cat]) if pd.notna(row[cat]) else 0
                )
            else:
                row_dict[template_key] = 0

        table_data.append(row_dict)

    # Create columns list for the table header, matching template logic
    columns = ['Message', 'Genre'] + [
        col.replace('_', ' ').title() for col in category_cols
    ]

    # Debug output for troubleshooting
    print(f"Debug: Prepared {len(table_data)} records for the table.")
    print(f"Debug: Columns for template = {columns[:5]}...")
    
    if table_data:
        sample_row = table_data[0]
        print(f"Debug: Sample row keys = {list(sample_row.keys())[:10]}...")
        
        message_content = sample_row.get('message', 'NOT FOUND')
        print(f"Debug: Message content = '{message_content[:100]}...'")
        print(f"Debug: Genre = '{sample_row.get('genre', 'NOT FOUND')}'")
        
        # Check a few category values using the template key
        for cat in category_cols[:3]:
            template_key = cat.lower().replace(' ', '_')
            value = sample_row.get(template_key, 'NOT FOUND')
            print(f"Debug: {cat} = {value}")

    # Create visualizations with error handling
    try:
        graphs = create_visualizations()
        graph_titles = [
            'Category Distribution',
            'Genre Distribution',
            'Category Correlations',
            'Message Length Distribution',
            'Top Words Frequency'
        ]
        # Generate unique IDs for HTML elements
        ids = [f'graph{i}' for i in range(len(graphs))]
    except Exception as e:
        logging.exception("Error creating visualizations")
        graphs = []
        graph_titles = []
        ids = []
        flash("Error loading visualizations")

    # Render template with all prepared data
    return render_template(
        'master.html',
        table_data=table_data,
        columns=columns,
        category_cols=category_cols,
        row_limit=limit,
        search_query=search_query,
        graphs=graphs,
        graph_titles=graph_titles,
        ids=ids
    )


@app.route('/simple')
def simple_view():
    """Simple view without complex visualizations for debugging.

    Provides a simplified interface for viewing and searching messages without
    the overhead of complex visualizations. Useful for debugging and performance
    testing when full dashboard functionality is not needed.

    Query Parameters:
        limit (int, optional): Maximum number of messages to display.
                              Defaults to 50.
        search (str, optional): Search query to filter messages.
                               Defaults to empty string.

    Returns:
        str: Rendered HTML page with message table and basic styling.

    Features:
        - Clean, responsive table layout
        - Search functionality
        - Message filtering and pagination
        - Category visualization with checkmarks
        - Direct link to message classification
    """
    # Get query parameters with defaults
    limit = request.args.get('limit', default=50, type=int)
    search_query = request.args.get('search', default='', type=str)

    # Check if data is available
    if df.empty:
        flash("No data available. Please check database connection.")
        return render_template_string(
            """
            <h1>Error</h1>
            <p>No data available. Please check database connection.</p>
            <p>Database path: {{ db_path }}</p>
            """,
            db_path=db_path
        )

    # Get all category columns by excluding known non-category ones
    category_cols = [
        col for col in df.columns
        if col not in ['id', 'message', 'original', 'genre']
    ]

    # Search and filter data based on parameters
    if search_query:
        df_filtered = search_messages(df, search_query, limit)
        if df_filtered.empty:
            flash(f"No messages found containing '{search_query}'")
            df_filtered = df.head(limit)
    else:
        df_filtered = df.head(limit) if limit > 0 else df

    # Prepare data for template rendering
    table_data = []
    for _, row in df_filtered.iterrows():
        row_data = {
            'message': str(row.get('message', 'No message')),
            'genre': str(row.get('genre', '-'))
        }

        # Add category data with proper handling of missing values
        for cat in category_cols:
            if cat in row:
                row_data[cat] = int(row[cat]) if pd.notna(row[cat]) else 0
            else:
                row_data[cat] = 0

        table_data.append(row_data)

    # Create column headers - ensure Message comes first for better UX
    columns = ['Message', 'Genre'] + [
        col.replace('_', ' ').title() for col in category_cols
    ]

    # Debug information for development
    print(f"Debug: Found {len(table_data)} records")
    print(f"Debug: Columns = {columns}")
    if table_data:
        print(f"Debug: First record = {table_data[0]}")

    # Use the improved template HTML with embedded CSS and responsive design
    template_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Disaster Response - Message Search</title>
        <style>
            /* Base styling for the page */
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            
            /* Main container styling */
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            /* Search section styling */
            .search-section {
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            
            /* Flexible search form layout */
            .search-form {
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            
            /* Search input field styling */
            .search-input {
                flex: 1;
                min-width: 300px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            
            /* Button base styling */
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                text-decoration: none;
                display: inline-block;
                text-align: center;
            }
            
            /* Button color variants */
            .btn-primary { background-color: #007bff; color: white; }
            .btn-secondary { background-color: #6c757d; color: white; }
            .btn:hover { opacity: 0.9; }

            /* Table container with horizontal scroll */
            .table-container {
                overflow-x: auto;
                margin-top: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }

            /* Responsive table styling */
            .responsive-table {
                width: 100%;
                border-collapse: collapse;
                margin: 0;
                background-color: white;
            }

            /* Table header styling with sticky positioning */
            .responsive-table th {
                background-color: #343a40;
                color: white;
                padding: 12px 8px;
                text-align: left;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
                border-bottom: 2px solid #dee2e6;
                font-size: 14px;
            }

            /* Table cell styling */
            .responsive-table td {
                padding: 10px 8px;
                border-bottom: 1px solid #dee2e6;
                vertical-align: top;
                font-size: 13px;
            }

            /* Alternating row colors */
            .responsive-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }

            /* Row hover effect */
            .responsive-table tr:hover {
                background-color: #e9ecef;
            }

            /* Message column specific styling */
            .message-cell {
                max-width: 350px;
                min-width: 250px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                white-space: pre-wrap;
                line-height: 1.4;
            }

            /* Genre column specific styling */
            .genre-cell {
                min-width: 80px;
                max-width: 100px;
                text-align: center;
                font-weight: 500;
            }

            /* Category column base styling */
            .category-cell {
                text-align: center;
                min-width: 60px;
                max-width: 80px;
                font-weight: bold;
            }

            /* Category true/false color coding */
            .category-true {
                background-color: #d4edda !important;
                color: #155724;
            }

            .category-false {
                background-color: #f8d7da !important;
                color: #721c24;
            }

            /* Alert message styling */
            .alert {
                padding: 12px;
                margin-bottom: 20px;
                border: 1px solid transparent;
                border-radius: 4px;
            }
            .alert-info {
                color: #0c5460;
                background-color: #d1ecf1;
                border-color: #bee5eb;
            }
            .alert-warning {
                color: #856404;
                background-color: #fff3cd;
                border-color: #ffeaa7;
            }

            /* Statistics display */
            .stats {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .stat-item {
                background-color: #e9ecef;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 14px;
            }

            /* Classification link styling */
            .classification-link {
                background-color: #28a745;
                color: white;
                padding: 12px 24px;
                border-radius: 4px;
                text-decoration: none;
                display: inline-block;
                margin-bottom: 20px;
                font-weight: 500;
            }
            .classification-link:hover {
                background-color: #218838;
                color: white;
                text-decoration: none;
            }

            /* Mobile responsive design */
            @media (max-width: 768px) {
                .message-cell {
                    max-width: 200px;
                    min-width: 150px;
                }
                .responsive-table th,
                .responsive-table td {
                    padding: 8px 4px;
                    font-size: 12px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Page header -->
            <h1>🚨 Disaster Response Pipeline</h1>
            <p>Analyze disaster messages and get emergency response 
               recommendations</p>

            <!-- Link to message classification feature -->
            <a href="/go" class="classification-link">
                🔍 Classify New Message
            </a>

            <!-- Search interface -->
            <div class="search-section">
                <h3>Search Messages</h3>
                <form method="GET" action="/simple" class="search-form">
                    <input type="text" name="search" 
                           placeholder="Search messages..."
                           value="{{ search_query }}" class="search-input">
                    <input type="number" name="limit" placeholder="Limit" 
                           min="1" max="1000" value="{{ row_limit }}" 
                           class="btn btn-secondary" style="width: 100px;">
                    <button type="submit" class="btn btn-primary">Search</button>
                    <a href="/simple" class="btn btn-secondary">Clear</a>
                    <a href="/" class="btn btn-secondary">Full Dashboard</a>
                </form>
            </div>

            <!-- Flash messages display -->
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <div class="alert alert-info">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}

            <!-- Statistics display -->
            <div class="stats">
                <div class="stat-item">
                    <strong>Total Messages:</strong> {{ table_data|length }}
                </div>
                {% if search_query %}
                <div class="stat-item">
                    <strong>Search:</strong> "{{ search_query }}"
                </div>
                {% endif %}
                <div class="stat-item">
                    <strong>Showing:</strong> {{ row_limit }} rows max
                </div>
            </div>

            <!-- Main data table -->
            <div class="table-container">
                {% if table_data %}
                <table class="responsive-table">
                    <thead>
                        <tr>
                            {% for column in columns %}
                            <th>{{ column }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in table_data %}
                        <tr>
                            <!-- Message content cell -->
                            <td class="message-cell">{{ row.message }}</td>
                            <!-- Genre cell -->
                            <td class="genre-cell">{{ row.genre }}</td>
                            <!-- Category cells with visual indicators -->
                            {% for cat in category_cols %}
                                {% if row[cat] == 1 %}
                                    <td class="category-cell category-true">
                                        ✓
                                    </td>
                                {% else %}
                                    <td class="category-cell category-false">
                                        ✗
                                    </td>
                                {% endif %}
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                    <div class="alert alert-warning">
                        No data to display. Please check your search criteria 
                        or database connection.
                    </div>
                {% endif %}
            </div>

            <!-- Legend for category symbols -->
            <div style="margin-top: 20px; padding: 15px; 
                        background-color: #f8f9fa; border-radius: 5px;">
                <h4>Legend:</h4>
                <p>
                    <span class="category-cell category-true" 
                          style="padding: 5px 10px; margin-right: 15px;">
                        ✓
                    </span> Category applies to message
                    <span class="category-cell category-false" 
                          style="padding: 5px 10px;">
                        ✗
                    </span> Category does not apply to message
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    # Render template with all prepared data
    return render_template_string(
        template_html,
        table_data=table_data,
        columns=columns,
        category_cols=category_cols,
        search_query=search_query,
        row_limit=limit
    )


@app.route('/go', methods=['GET', 'POST'])
def go():
    """Handle message classification and feedback with word analysis.

    Provides the main interface for classifying disaster messages using ML
    model and analyzing word usage patterns in the dataset. Handles both GET
    requests for displaying the form and POST requests for processing feedback.

    POST Parameters (for feedback):
        feedback (str): User feedback on prediction accuracy
                       ('accurate' or other)
        message (str): Original message that was classified
        prediction (str): The prediction result that user is providing
                         feedback on

    GET Parameters (for classification):
        query (str): The message text to classify and analyze

    Returns:
        str: Rendered HTML template with classification results and word
             analysis.

    Template Variables:
        query (str): The input message
        classification_result (dict): Category predictions from ML model
        recommendations (list): Emergency response recommendations
        word_analysis (dict): Statistical analysis of word usage in dataset
        word_graphs (list): JSON-encoded visualizations for word analysis
        word_graph_titles (list): Titles for word analysis graphs
        word_ids (list): HTML element IDs for word graphs
        error (str): Error message if classification fails

    Features:
        - ML-based message classification
        - Emergency response recommendations
        - Word frequency analysis
        - Dataset statistics and visualizations
        - User feedback collection
    """
    # Handle POST request for feedback submission
    if request.method == 'POST':
        feedback = request.form.get('feedback')
        if feedback:
            # Create new feedback record in database
            new_feedback = Feedback(
                message=request.form.get('message', ''),
                prediction=request.form.get('prediction', ''),
                is_correct=(feedback == 'accurate')
            )
            db.session.add(new_feedback)
            db.session.commit()
            flash("Thank you! Your feedback has been submitted.")
            return redirect('/')

    # Get query parameter for message classification
    query = request.args.get('query', '').strip()

    # If no query provided, show the classification form
    if not query:
        return render_template('go.html', query='')

    print(f"Debug: Received query: '{query}'")

    # Check if model is available for classification
    if model is None:
        return render_template(
            'go.html',
            error="Model is not available. Please check model file and "
                  "try again."
        )

    try:
        # Make prediction using the loaded ML model
        pred = model.predict([query])[0]

        # Get category names from DataFrame columns
        category_cols = [
            col for col in df.columns
            if col not in ['id', 'message', 'original', 'genre']
        ]

        # Create results dictionary mapping categories to predictions
        results = dict(zip(category_cols, map(int, pred)))

        # Filter for positive results to generate recommendations
        positive_results = {cat: val for cat, val in results.items() 
                          if val == 1}

        # Get recommendations for positive categories
        recommendations = []
        for cat in positive_results.keys():
            if cat in RECOMMENDATIONS:
                recommendations.append(RECOMMENDATIONS[cat])

        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations.append({
                'text': 'Contact local emergency services and relief '
                        'organizations',
                'links': [
                    {
                        'name': 'UN Emergency Relief',
                        'url': 'https://www.un.org/en/humanitarian-response'
                    }
                ]
            })

        # Perform word analysis on the input query
        word_analysis = analyze_word_in_dataset(query, df)
        word_visualizations = []
        word_viz_titles = []
        word_viz_ids = []

        # Create visualizations if word analysis found matches
        if word_analysis and word_analysis['message_count'] > 0:
            # Get messages containing the word from dataset
            mask = df['message'].str.contains(
                query.lower(), case=False, na=False
            )
            word_messages = df[mask]

            # Create basic visualizations for word analysis
            basic_visuals = create_word_visualizations(word_analysis)
            word_visualizations.extend(basic_visuals)

            # Define titles for basic visualizations
            basic_titles = [
                f'Categories for "{query.title()}" Messages',
                f'Message Sources for "{query.title()}"',
                f'Related Words for "{query.title()}"',
                f'Usage Statistics for "{query.title()}"'
            ]
            word_viz_titles.extend(basic_titles)

            # Create advanced visualizations for deeper analysis
            advanced_visuals = create_advanced_word_visualizations(
                query, df, word_messages
            )
            word_visualizations.extend(advanced_visuals)

            # Define titles for advanced visualizations
            advanced_titles = [
                f'Category Comparison Analysis',
                f'Message Complexity Distribution',
                f'Category Relationships',
                f'Message Length Comparison',
                f'Language Distribution',
                f'Common Category Combinations',
                f'Natural Disaster Types'
            ]
            word_viz_titles.extend(advanced_titles)

            # Generate insights from word analysis
            word_insights = get_word_insights(query, df, word_messages)
            word_analysis['insights'] = word_insights

        # Generate unique IDs for HTML elements
        word_viz_ids = [
            f'word_graph_{i}' for i in range(len(word_visualizations))
        ]

        # Render template with all analysis results
        return render_template(
            'go.html',
            query=query,
            classification_result=results,
            recommendations=recommendations,
            word_analysis=word_analysis,
            word_graphs=word_visualizations,
            word_graph_titles=word_viz_titles,
            word_ids=word_viz_ids
        )

    except Exception as e:
        # Log full exception details for debugging
        logging.exception("Error during classification")
        return render_template(
            'go.html',
            error=f"Error processing request: {str(e)}"
        )


@app.route('/search_api')
def search_api():
    """API endpoint for live search suggestions.
    
    Provides real-time search suggestions based on message content. Searches through
    the loaded DataFrame for messages containing the query string and returns a 
    limited number of truncated results.
    
    Query Parameters:
        q (str): Search query string. Must be at least 2 characters long.
        limit (int, optional): Maximum number of results to return. Defaults to 10.
    
    Returns:
        flask.Response: JSON response containing a list of truncated message strings
                       that match the search query. Returns empty list if no matches
                       found or if query is invalid.
    
    Note:
        - Returns empty list if DataFrame is empty
        - Query must be at least 2 characters long
        - Search is case-insensitive
        - Results are truncated to 100 characters for display
    """
    # Return empty results if DataFrame is not loaded or empty
    if df.empty:
        return jsonify([])
    
    # Extract and validate query parameters
    query = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    
    # Validate query length - require minimum 2 characters
    if not query or len(query) < 2:
        return jsonify([])
    
    # Search for messages containing the query (case-insensitive)
    mask = df['message'].str.contains(query, case=False, na=False)
    results = df[mask]['message'].head(limit).tolist()
    
    # Return truncated results for better display formatting
    return jsonify([truncate_text(msg, 100) for msg in results])


# ================== Run ==================
if __name__ == '__main__':
    """Main execution block for the Flask application.
    
    Initializes the Flask app context, creates database tables, loads visualizations,
    and starts the development server. Includes error handling for visualization
    loading and provides status feedback during startup.
    """
    # Initialize Flask application context for database operations
    with app.app_context():
        # Create all database tables based on defined models
        db.create_all()
        print("Starting Disaster Response Flask App...")
        print("Loading model and creating visualizations...")
        
        # Warm up the cache by pre-loading visualizations
        try:
            # Attempt to create and cache visualizations for faster initial load
            create_visualizations()
            print("✅ Visualizations loaded successfully")
        except Exception as e:
            # Log visualization errors but continue app startup
            print(f"⚠️ Error loading visualizations: {e}")
    
    # Start Flask development server
    # host='0.0.0.0' allows external connections
    # port=3001 specifies the listening port
    # debug=True enables auto-reload and detailed error messages
    app.run(host='0.0.0.0', port=3001, debug=True)
