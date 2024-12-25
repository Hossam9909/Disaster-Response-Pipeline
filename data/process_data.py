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
