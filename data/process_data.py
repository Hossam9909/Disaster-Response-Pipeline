import sys
import logging
import os
import re
import string
import multiprocessing
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from joblib import Parallel, delayed


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure NLTK resources
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def configure_logging():
    """
    Configure logging for the application.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


configure_logging()


def validate_input_files(filepaths):
    """
    Validate the input filepaths to ensure they exist and are readable.

    Args:
    - filepaths (list): List of filepaths to validate.

    Returns:
    - None
    """
    for filepath in filepaths:
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            sys.exit(1)
        if not filepath.endswith('.csv'):
            logging.error(f"Invalid file format (expected .csv): {filepath}")
            sys.exit(1)


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    Args:
    - messages_filepath (str): Filepath to the messages dataset.
    - categories_filepath (str): Filepath to the categories dataset.
    Returns:
    - df (DataFrame): Merged DataFrame of messages and categories.
    """
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        df = messages.merge(categories, on='id')
        logging.info("Data successfully loaded and merged.")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)


def tokenize(text):
    """
    Tokenize and preprocess text by normalizing, removing punctuation, and lemmatizing.

    Args:
    - text (str): Input text.

    Returns:
    - tokens (list): List of processed tokens.
    """
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token)
              for token in tokens if token not in stop_words]
    return tokens


def clean_data(df):
    """
    Clean the merged DataFrame of messages and categories.
    Args:
    - df (DataFrame): Merged DataFrame of messages and categories.
    Returns:
    - df (DataFrame): Cleaned DataFrame.
    """
    try:
        # Split the categories column into separate columns
        categories = df['categories'].str.split(';', expand=True)

        # Extract new column names from the first row
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x.split('-')[0])
        categories.columns = category_colnames

        # Convert category values to binary (0 or 1)
        for column in categories:
            categories[column] = categories[column].astype(str).str[-1]
            categories[column] = pd.to_numeric(categories[column], errors='coerce').fillna(0).astype(int)
            # Ensure all values are 0 or 1
            categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)

        # Drop the original categories column from df
        df = df.drop('categories', axis=1)

        # Concatenate the original DataFrame with the new category columns
        df = pd.concat([df, categories], axis=1)

        # Remove duplicates
        initial_length = len(df)
        df = df.drop_duplicates()
        final_length = len(df)
        logging.info(f"Removed {initial_length - final_length} duplicate rows.")

        return df

    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        sys.exit(1)


def save_cleaned_data(df, database_filepath):
    """
    Save the cleaned DataFrame into a SQLite database using sqlite3 directly.
    
    Args:
    - df (DataFrame): Cleaned DataFrame.
    - database_filepath (str): Filepath for the SQLite database.
    """
    try:
        conn = sqlite3.connect(database_filepath)
        df.to_sql('DisasterResponse', conn, index=False, if_exists='replace')
        conn.close()
        logging.info(f"Cleaned data saved to database at {database_filepath}.")
    except Exception as e:
        logging.error(f"Error saving data to database: {e}")
        sys.exit(1)


def summarize_data(df):
    """
    Print a summary report of the dataset.

    Args:
    - df (DataFrame): The DataFrame to summarize.
    """
    try:
        logging.info(f"Dataset contains {df.shape[0]} rows and {
                     df.shape[1]} columns.")
        logging.info(f"Columns: {', '.join(df.columns)}")
        logging.info(f"Number of duplicate rows: {df.duplicated().sum()}")
    except Exception as e:
        logging.error(f"Error summarizing data: {e}")


def main():
    """
    Main function to execute the ETL pipeline.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        validate_input_files([messages_filepath, categories_filepath])

        logging.info(f'Loading data...\n    MESSAGES: {
                     messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        logging.info('Cleaning data...')
        df = clean_data(df)

        summarize_data(df)

        logging.info(f'Saving data...\n    DATABASE: {database_filepath}')
        save_cleaned_data(df, database_filepath)

        logging.info('Cleaned data saved to database successfully!')
    else:
        logging.error(
            "Please provide the filepaths of messages and categories as the first "
            "and second arguments, and the database filepath as the third.\n"
            "Example: python process_data.py data/disaster_messages.csv "
            "data/disaster_categories.csv data/DisasterResponse.db"
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
