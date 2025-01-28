import sys
import logging
import os
import re
import string
import multiprocessing
import pandas as pd
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


def clean_row(row):
    """
    Clean a single row of the DataFrame.
    Args:
    - row (Series): A row of the DataFrame.
    Returns:
    - row (Series): Cleaned row.
    """
    try:
        # Split categories into separate columns
        categories_split = row['categories'].str.split(';', expand=True)
        # Extract column names
        category_colnames = categories_split.iloc[0].apply(lambda x: x[:-2])
        categories_split.columns = category_colnames

        # Convert category values to binary
        for column in categories_split:
            categories_split[column] = categories_split[column].str[-1].astype(
                int)
            if not categories_split[column].isin([0, 1]).all():
                logging.warning(
                    f"Non-binary values found in column {column}. They will be corrected.")
                categories_split[column] = categories_split[column].apply(
                    lambda x: 1 if x > 0 else 0)

        # Replace categories column with the new category columns
        row = row.drop('categories')
        row = pd.concat([row, categories_split], axis=0)
        return row
    except Exception as e:
        logging.error(f"Error cleaning row: {e}")
        return row


def clean_data(df):
    """
    Clean the merged DataFrame using parallel processing.
    Args:
    - df (DataFrame): Merged DataFrame of messages and categories.
    Returns:
    - df (DataFrame): Cleaned DataFrame.
    """
    try:
        # Use multiprocessing to clean rows in parallel
        num_cores = multiprocessing.cpu_count()
        cleaned_data = Parallel(n_jobs=num_cores)(
            delayed(clean_row)(row) for _, row in df.iterrows())
        df = pd.DataFrame(cleaned_data)

        # Remove duplicates
        initial_length = len(df)
        df = df.drop_duplicates()
        final_length = len(df)
        logging.info(f"Removed {initial_length -
                     final_length} duplicate rows.")

        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        sys.exit(1)


def save_cleaned_data(df, database_filepath):
    """
    Save the cleaned DataFrame into a SQLite database.
    Args:
    - df (DataFrame): Cleaned DataFrame.
    - database_filepath (str): Filepath for the SQLite database.
    """
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
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
        logging.error('Please provide the filepaths of the messages and categories datasets as the first and second argument '
                      'and the filepath of the database to save the cleaned data to as the third argument. \n\nExample: python '
                      'process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db')


if __name__ == '__main__':
    main()
