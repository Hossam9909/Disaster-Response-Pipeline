"""Disaster Response ETL Pipeline.

This module contains an ETL (Extract, Transform, Load) pipeline for processing
disaster response messages and categories data. The pipeline merges datasets,
cleans and transforms the data, and stores the results in a SQLite database
for use in machine learning applications.

The pipeline handles:
- Loading and merging messages and categories CSV files
- Parsing and cleaning category labels into binary format
- Text preprocessing and tokenization
- Data validation and duplicate removal
- Database storage with SQLite

Typical usage example:
    python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
"""

import sys
import logging
import os
import re
import string
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available
import nltk


def ensure_nltk_data():
    """Download NLTK data if not already present."""
    import os
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
                print(
                    f"Warning: Could not download NLTK data '{download_name}': {e}")


# Call once at module level
ensure_nltk_data()


def configure_logging():
    """Configure logging settings for the application.

    Sets up logging with INFO level and a standardized format that includes
    timestamps, log levels, and messages. This provides visibility into
    the ETL process execution and helps with debugging.

    Side Effects:
        Configures the global logging settings for the entire application.
        All subsequent logging calls will use this configuration.

    Note:
        This function should be called once at the start of the application
        to establish consistent logging behavior throughout the pipeline.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize logging configuration at module load time
configure_logging()


def validate_input_files(filepaths):
    """Validate input file paths to ensure they exist and have correct format.

    Performs comprehensive validation of input files including existence checks
    and format validation. This prevents the pipeline from proceeding with
    invalid inputs and provides clear error messages for troubleshooting.

    Args:
        filepaths (list): List of file paths to validate. Each path should
                         point to a readable CSV file.

    Raises:
        SystemExit: If any file doesn't exist or has incorrect format.
                   Exits with code 1 and logs appropriate error messages.

    Side Effects:
        Logs error messages and terminates the program if validation fails.

    Note:
        This function performs fail-fast validation to catch input issues
        early in the pipeline execution before any processing begins.
    """
    for filepath in filepaths:
        # Check if file exists and is accessible
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            sys.exit(1)

        # Validate file format (must be CSV)
        if not filepath.endswith('.csv'):
            logging.error(f"Invalid file format (expected .csv): {filepath}")
            sys.exit(1)


def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets.

    Loads two CSV files containing disaster response messages and their
    corresponding categories, then merges them on the 'id' column to create
    a unified dataset for processing.

    Args:
        messages_filepath (str): File path to the CSV containing disaster messages.
                               Expected to have columns: id, message, original, genre.
        categories_filepath (str): File path to the CSV containing message categories.
                                 Expected to have columns: id, categories.

    Returns:
        pd.DataFrame: Merged DataFrame containing both messages and categories
                     data, joined on the 'id' column.

    Raises:
        SystemExit: If files cannot be loaded or merged successfully.
                   Logs appropriate error messages before terminating.

    Side Effects:
        Logs successful completion of data loading and merging operation.

    Note:
        The merge operation uses an inner join on 'id', so only records
        with matching IDs in both files will be included in the result.
    """
    try:
        # Load individual CSV files into DataFrames
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)

        # Merge DataFrames on the common 'id' column
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
    """Tokenize and preprocess text for natural language processing.

    Performs comprehensive text preprocessing including normalization,
    punctuation removal, tokenization, stop word filtering, and lemmatization.
    This standardized preprocessing ensures consistent text representation
    for downstream machine learning tasks.

    Args:
        text (str): Raw input text to be tokenized and preprocessed.

    Returns:
        list: List of processed tokens (strings) with stop words removed,
              punctuation cleaned, and words lemmatized to their root forms.

    Example:
        >>> tokenize("Hello! How are you doing today?")
        ['hello', 'today']

    Note:
        This function uses NLTK's English stop words and WordNet lemmatizer.
        The preprocessing steps should match those used in the ML training
        pipeline for consistency.
    """
    # Convert text to lowercase for normalization
    text = text.lower()

    # Remove all punctuation characters using regex
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation

    # Tokenize the cleaned text into individual words
    tokens = word_tokenize(text)

    # Initialize lemmatizer for word root extraction
    lemmatizer = WordNetLemmatizer()

    # Load English stop words for filtering
    stop_words = set(stopwords.words('english'))

    # Apply lemmatization and stop word removal
    tokens = [lemmatizer.lemmatize(token)
              for token in tokens if token not in stop_words]

    return tokens


def clean_data(df):
    """Clean and transform the merged DataFrame of messages and categories.

    Performs comprehensive data cleaning including:
    - Parsing the semicolon-separated categories column into individual binary columns
    - Converting category values to proper binary format (0 or 1)
    - Handling invalid values and ensuring data quality
    - Removing duplicate records
    - Data type standardization

    Args:
        df (pd.DataFrame): Merged DataFrame containing messages and categories
                          data with a 'categories' column in format 'cat1-0;cat2-1;...'

    Returns:
        pd.DataFrame: Cleaned DataFrame with individual binary columns for each
                     category, duplicates removed, and proper data types.

    Raises:
        SystemExit: If data cleaning fails due to unexpected data format
                   or processing errors. Logs error details before terminating.

    Side Effects:
        Logs information about duplicate removal and data transformation progress.

    Note:
        The categories column format is expected to be: 'category1-0;category2-1;...'
        where each category has a binary value (0 or 1) separated by a hyphen.
    """
    try:
        # Split the categories column into separate columns based on semicolon delimiter
        categories = df['categories'].str.split(';', expand=True)

        # Extract category names from the first row by splitting on hyphen and taking first part
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x.split('-')[0])
        categories.columns = category_colnames

        # Convert category values to proper binary format (0 or 1)
        for column in categories:
            # Extract the numeric value (last character after the hyphen)
            categories[column] = categories[column].astype(str).str[-1]

            # Convert to numeric, handling any conversion errors by filling with 0
            categories[column] = pd.to_numeric(
                categories[column], errors='coerce').fillna(0).astype(int)

            # Ensure all values are binary (convert any value > 1 to 1)
            categories[column] = categories[column].apply(
                lambda x: 1 if x > 1 else x)

        # Remove the original categories column from the main DataFrame
        df = df.drop('categories', axis=1)

        # Concatenate the original DataFrame with the new binary category columns
        df = pd.concat([df, categories], axis=1)

        # Remove duplicate records and log the cleanup results
        initial_length = len(df)
        df = df.drop_duplicates()
        final_length = len(df)
        logging.info(
            f"Removed {initial_length - final_length} duplicate rows.")

        return df

    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        sys.exit(1)


def save_cleaned_data(df, database_filepath):
    """Save the cleaned DataFrame to a SQLite database.

    Stores the processed DataFrame in a SQLite database table named 
    'DisasterResponse'. Uses direct sqlite3 connection for reliable
    database operations and replaces any existing table with the same name.

    Args:
        df (pd.DataFrame): Cleaned DataFrame containing processed messages
                          and binary category columns ready for ML training.
        database_filepath (str): File path where the SQLite database should
                               be created or updated. Will create the file
                               if it doesn't exist.

    Raises:
        SystemExit: If database operations fail due to permissions, disk space,
                   or other database-related errors. Logs error details.

    Side Effects:
        Creates or overwrites the specified SQLite database file.
        Logs successful completion of database save operation.

    Note:
        The table name 'DisasterResponse' is hardcoded and expected by
        downstream ML pipeline components. The 'replace' option ensures
        clean data loading on subsequent runs.
    """
    try:
        # Establish connection to SQLite database (creates file if doesn't exist)
        conn = sqlite3.connect(database_filepath)

        # Save DataFrame to database table, replacing any existing data
        df.to_sql('DisasterResponse', conn, index=False, if_exists='replace')

        # Close database connection to free resources
        conn.close()

        logging.info(f"Cleaned data saved to database at {database_filepath}.")

    except Exception as e:
        logging.error(f"Error saving data to database: {e}")
        sys.exit(1)


def summarize_data(df):
    """Generate and log a comprehensive summary report of the dataset.

    Provides detailed information about the dataset structure and quality
    including dimensions, column names, and data quality metrics. This
    summary helps validate the ETL process results and identify potential
    issues before proceeding to machine learning.

    Args:
        df (pd.DataFrame): The DataFrame to analyze and summarize.
                          Should be the final cleaned dataset.

    Side Effects:
        Logs dataset summary information including:
        - Dataset dimensions (rows and columns)
        - Complete list of column names
        - Count of duplicate rows remaining

    Note:
        This function is primarily for monitoring and validation purposes.
        It helps ensure the ETL process completed successfully and the
        data is in the expected format for downstream processing.
    """
    try:
        # Log basic dataset dimensions
        logging.info(
            f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        # Log all column names for validation
        logging.info(f"Columns: {', '.join(df.columns)}")

        # Check for any remaining duplicates (should be 0 after cleaning)
        logging.info(f"Number of duplicate rows: {df.duplicated().sum()}")

    except Exception as e:
        logging.error(f"Error summarizing data: {e}")


def main():
    """Main function to execute the complete ETL pipeline.

    Orchestrates the entire ETL workflow from command-line argument validation
    through data loading, cleaning, and database storage. Provides comprehensive
    error handling and logging throughout the process.

    Command Line Args:
        messages_filepath (str): Path to CSV file containing disaster messages.
        categories_filepath (str): Path to CSV file containing message categories.
        database_filepath (str): Path where SQLite database should be created.

    Steps:
        1. Validate command line arguments (exactly 3 required)
        2. Validate input file existence and format
        3. Load and merge the input CSV files
        4. Clean and transform the merged data
        5. Generate data quality summary
        6. Save cleaned data to SQLite database

    Example:
        python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    Raises:
        SystemExit: If incorrect arguments provided or any step in the pipeline fails.
                   Logs appropriate error messages for troubleshooting.

    Side Effects:
        Creates SQLite database file with cleaned data.
        Logs progress and completion status throughout execution.
    """
    # Validate command line arguments - exactly 4 arguments expected (including script name)
    if len(sys.argv) == 4:
        # Extract command line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Step 1: Validate input files exist and have correct format
        validate_input_files([messages_filepath, categories_filepath])

        # Step 2: Load and merge the input datasets
        logging.info(f'Loading data...\n    MESSAGES: {
                     messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        # Step 3: Clean and transform the merged data
        logging.info('Cleaning data...')
        df = clean_data(df)

        # Step 4: Generate summary report for validation
        summarize_data(df)

        # Step 5: Save cleaned data to database
        logging.info(f'Saving data...\n    DATABASE: {database_filepath}')
        save_cleaned_data(df, database_filepath)

        # Log successful completion
        logging.info('Cleaned data saved to database successfully!')

    else:
        # Provide usage instructions for incorrect argument count
        logging.error("""Please provide the filepaths of messages and categories as the first and second arguments, and the database filepath as the third.

Example: python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db""")
        sys.exit(1)


# Execute main function when script is run directly
if __name__ == '__main__':
    main()
