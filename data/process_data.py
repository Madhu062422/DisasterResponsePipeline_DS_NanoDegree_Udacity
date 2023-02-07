import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    
    """
    @Desc: This function loads and merges data from message and categories csv files into one dataframe
    
    @Params:
        messages_filepath : path to csv messages file
        categories_filepath : path to csv categories file
    @Returns:
        df : combined dataframe containing both messages and categories
    """
    
    # loads data from messages.csv to msgs and categories.csv to categ
    msgs = pd.read_csv(messages_filepath)
    categ = pd.read_csv(categories_filepath)
    # merge messages and categories into df on 'id'
    df = pd.merge(msgs, categ, on="id")
    return df

def clean_data(df):
    
    """
    @Desc : Function to cleanse data
    
    @Params:
        df :  merged dataframe containing both category and messages
        
    @Returns:
        df -> cleaned df
    """
    
    # splitting values in categories column by ';'
    categories_df = df['categories'].str.split(pat=';',expand=True)
    
    #creating a dataframe with each split values in categories as column names
    name = categories_df.iloc[[1]]
    cat_col_name = [c.split('-')[0] for c in name.values[0]]
    categories_df.columns = cat_col_name
    
    for col in categories_df:
        categories_df[col] = categories_df[col].str[-1]
        categories_df[col] = categories_df[col].astype(np.int)
    
    #print("1.df columns ",df.columns)
    #drop categories column from df
    df = df.drop('categories', axis=1)
    #concatenate categories to combined and cleansed df
    df = pd.concat([df,categories_df], axis=1)
    #drop duplictaes if any
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    
    """
    @Desc: Function to save cleaned data into SQLite Database
    
    @Params:
        cleansed dataframe :  df
        database_filename : path of sql
        
    @Returns: None 
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')  


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