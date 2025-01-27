import sys
import logging
import pandas as pd
from sqlalchemy import create_engine


def configure_logging():
    """
    Configure logging for the application.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


configure_logging()


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


def clean_data(df):
    """
    Clean the merged DataFrame.
    Args:
    - df (DataFrame): Merged DataFrame of messages and categories.
    Returns:
    - df (DataFrame): Cleaned DataFrame.
    """
    try:
        # Split categories into separate columns
        categories_split = df['categories'].str.split(';', expand=True)
        # Extract column names
        row = categories_split.iloc[0]
        category_colnames = row.apply(lambda x: x[:-2])
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
        df = df.drop('categories', axis=1)
        df = pd.concat([df, categories_split], axis=1)

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


def main():
    """
    Main function to execute the ETL pipeline.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info(f'Loading data...\n    MESSAGES: {
                     messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        logging.info('Cleaning data...')
        df = clean_data(df)

        logging.info(f'Saving data...\n    DATABASE: {database_filepath}')
        save_cleaned_data(df, database_filepath)

        logging.info('Cleaned data saved to database successfully!')
    else:
        logging.error('Please provide the filepaths of the messages and categories datasets as the first and second argument '
                      'and the filepath of the database to save the cleaned data to as the third argument. \n\nExample: python '
                      'process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db')


if __name__ == '__main__':
    main()
