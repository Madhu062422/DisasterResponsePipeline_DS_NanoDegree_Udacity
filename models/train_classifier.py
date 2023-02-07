"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data_from_db(database_filepath):
    
    """
    @Desc: Function to load data from the SQLite DB
    
    @Params:
        database_filepath : Path to SQLite database
    @Returns:
        X : a dataframe containing features
        Y : a dataframe containing labels
        category_names : categories name in a list
    """
    
    #loading data from sqlite DB and storing it into df
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(str('data/'+table_name),engine)
    
    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)
    
    # Replacing 2 with 1 (majority class) in related field to consider it a valid response (as it could be an error).
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # getting data for modelling 
    #Input data 
    X = df['message']
    #Tagget data
    y = df.iloc[:,4:]
    
    # categories name in a list
    category_names = y.columns #visualization purpose
    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    @Desc:  Function to tokenize the text
    
    @Params: 
    text :  Text message to be tokenized
    
    @Returns: 
    clean_tokens : List of tokens extracted from text provided
    """
        
    # Lets replace all urls with urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urlplaceholder = 'urlplaceholder'
    
    # find urls from given text 
    urls_identified = re.findall(url_regex, text)
    
    # replacing urls with placeholders
    for urls in urls_identified:
        text = text.replace(urls, urlplaceholder)

    # using nltk word_tokenize lets extract the word tokens from the given text
    tokens = nltk.word_tokenize(text)
    
    # using nltk lammatizer, covert the tokens to its verb form
    lemmatizer = nltk.WordNetLemmatizer()

    # getting a list of tokenized text
    tokens_tokenized = [lemmatizer.lemmatize(x).lower().strip() for x in tokens]
    
    return tokens_tokenized
    
# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extracts the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        
        '''
        @Desc: Function to extarct starting verb from the given text
        @Params: text
        @Returns : Boolean 
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)    

def build_pipeline():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    @Desc: Function to evaluate model performance

    @Params: 
            ML Pipeline, 
            X_test : Features set aside for testing, 
            Y_test :  corresponding labels set aside for testing, 
            category_names : list of category names (labels)

     @Returns : model accuracy, classification report
    """
    Y_pred = pipeline.predict(X_test)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))
   

def save_model_as_pickle(pipeline, pickle_filepath):
    
    """
        @Desc: Function to save model as pickle file after being trained
        
        @Params: 
            ML Pipeline, 
            pickle_filepath : path to save .pkl file
        
    """
 
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))

def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
Arguments Description: \n\
1) Path to SQLite destination database (e.g. disaster_response_db.db)\n\
2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")

if __name__ == '__main__':
    main()