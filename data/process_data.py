import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files and merge message and category data into a single dataframe.

    Args:
        messages_filepath (str): Filepath to the CSV file containing message data.
        categories_filepath (str): Filepath to the CSV file containing category data.

    Returns:
        df : A DataFrame containing merged data from both files.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on=['id'])
    return df


def clean_data(df):
    """
    Clean and preprocess a DataFrame containing message and category data.

    This function performs the following tasks:
    1. Splits the 'categories' column into 36 individual category columns.
    2. Renames the category columns based on the first row values.
    3. Converts category values to binary (0 or 1).
    4. Replaces the 'categories' column with the new category columns.
    5. Removes duplicate rows.

    Args:
        df: The input DataFrame containing message and category data.

    Returns:
        df: A cleaned DataFrame with individual category columns and no duplicates.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = list(categories.loc[0])
    category_colnames = [col.split('-')[0] for col in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers.
    get_last = lambda x: int(x[-1])
    for column in categories:
        categories[column] = categories[column].apply(get_last)
        
    # Removes rows whose category values are not 0 or 1
    get_binary = lambda x: x if x in [0, 1] else np.nan
    for col in categories.columns:
        categories[col] = categories[col].apply(get_binary)
    categories.dropna(inplace=True)

    # Replace `categories` column in `df` with new category columns.
    df.drop(columns = ['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates.
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save a DataFrame to an SQLite database.
    Args:
        df: The DataFrame to be saved to the database.
        database_filename (str): The filename for the SQLite database.

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('CleanData', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()