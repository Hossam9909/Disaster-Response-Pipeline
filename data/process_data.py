import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    Args:
    - messages_filepath (str): Filepath to the messages dataset.
    - categories_filepath (str): Filepath to the categories dataset.
    Returns:
    - df (DataFrame): Merged DataFrame of messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged DataFrame.
    Args:
    - df (DataFrame): Merged DataFrame of messages and categories.
    Returns:
    - df (DataFrame): Cleaned DataFrame.
    """
    # Split categories into separate columns
    categories_split = df['categories'].str.split(';', expand=True)
    # Extract column names
    row = categories_split.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories_split.columns = category_colnames

    # Convert category values to binary
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1].astype(int)

    # Replace categories column with the new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories_split], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_cleaned_data(df, database_filepath):
    """
    Save the cleaned DataFrame into a SQLite database.
    Args:
    - df (DataFrame): Cleaned DataFrame.
    - database_filepath (str): Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    """
    Main function to execute the ETL pipeline.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(
            messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_cleaned_data(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories datasets as the first and second argument '
              'and the filepath of the database to save the cleaned data to as the third argument. \n\nExample: python '
              'process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db')


if __name__ == '__main__':
    main()
