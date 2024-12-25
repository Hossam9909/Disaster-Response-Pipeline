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
