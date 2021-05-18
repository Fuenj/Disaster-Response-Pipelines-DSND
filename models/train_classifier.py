# import libraries
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
import pickle

def load_data(database_filepath):
    """ Function that loads a database table and return values, labels and category names
    Args:
        database_filepath:string. Location of the SQLite database.
    Returns:
        X: The training Data.
        Y: The training labels.
        category_names: Names used for data visualization.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """Tokenize text
    Args:
        text:string. Text to tokenize.
    Returns:
        clean_tokens:list. The tokenized text.
    """
  
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # use the word tokenizer to convert text into tokens
    tokens = word_tokenize(text)
    #initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    #Lemmatize and strip
    clean_tokens = []
    for tok in tokens:
    # Using lemmatization to strip all the words
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """This class extract the starting verb of a sentence,
       creating a new feature for the ML classifier.
       (reference:Udacity classroom)
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """Build and optimize model
    
    Args:
        None.
        
    Returns:
       None.
    """
    
    # build pipeline
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

   # set parameters for grid search 
    parameters = {
         'clf__estimator__n_estimators': [50, 100]
    }
    
    # optimize model
    
    model = GridSearchCV(model, param_grid=parameters)
    
    return model
    
def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model performance
    Args:
        model: ML Pipeline
        X_test: test data
        Y_test: test labels
        category_names: the category names
    """
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        category = category_names[i]
        print(category)
        print(classification_report(Y_test[category], Y_pred[:, i]))


def save_model(model, model_filepath):
    """Save model as pickle file
    Args:
        model: optimized classifier.
        model_filepath: path to save the pickle file.
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))



def main():
    """-Extract data from SQLite database;
       -Train ML model;
       -Estimate model;
       -Save trained model as Pickle.
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()