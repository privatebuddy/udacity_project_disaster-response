import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Functions:
            load data to pandas dataframe

        Arguments:
            messages_filepath: path to messages csv file
            categories_filepath: path to categories csv file
        Output:
            df: Loaded data as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
        Functions:
            clean data
                1. split categories by ;
                2. convert categories data to number instead of text
                3. drop duplicate data
        Arguments:
                df: data frame of merge data
        Output:
                clean_df: clean data frame ready to use
    """

    # split category data by ;
    categories = df['categories'].str.split(';', expand=True)

    # get first row of data
    first_row_data = categories[0:1]

    # apply lambda function to extract columns name
    category_column_names = first_row_data.apply(lambda x: x.str[:-2]).values[0]

    # set columns name
    categories.columns = category_column_names

    # convert to number
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    categories = categories[categories.related != 2]
    # drop old categories
    df.drop(['categories'], axis=1, inplace=True)

    # merge new categories
    df = pd.concat([df, categories], axis=1)

    # remove duplicate
    clean_df = df.drop_duplicates()

    return clean_df


def save_data(df, database_filename):
    """
        Functions:
            covert data frame to sqlite database file
        Arguments:
            df: data frame to convert to sqlite database
            database_filename: database name
        Output:
            save database ready to use
    """
    sqlite_path = "sqlite:///{0}".format(database_filename)
    engine = create_engine(sqlite_path)
    df.to_sql('message', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
