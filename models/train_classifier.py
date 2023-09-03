import sys
import pickle

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import re 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data from an SQLite database and prepare it for machine learning.

    Args:
        database_filepath (str): The filename of the SQLite database.

    Returns:
        tuple: A tuple containing the following elements:
            - X (pandas.Series): The message data.
            - y (pandas.DataFrame): The category data.
            - category_names (list): List of category column names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='CleanData', con=engine)
    X = df['message']
    y = df.drop(columns=['id','message', 'original', 'genre'])
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Tokenize and preprocess a given text.
    This function performs the following tasks:
    1. Converts the text to lowercase.
    2. Removes non-alphanumeric characters and replaces them with spaces.
    3. Tokenizes the text into words.
    4. Removes punctuation.
    5. Removes common English stopwords.
    6. Lemmatizes the tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of preprocessed and tokenized words.
    """
    # Tokenize the text into words
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    tokens = word_tokenize(text)

    # Remove punctuation 
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def build_model():
    """
    Build a machine learning pipeline for multi-output classification.

    This function creates a scikit-learn pipeline that includes the following steps:
    1. Tokenization using the `tokenize` function.
    2. Vectorization using CountVectorizer.
    3. TF-IDF transformation.
    4. Multi-output classification using RandomForestClassifier.

    Returns:
        pipelone (sklearn.pipeline.Pipeline): A machine learning pipeline for multi-output classification.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters = {
    #     'vect__ngram_range': ((1, 1), (1, 2)),
    #     'mclf__estimator__n_estimators': [50, 100],
    #     'mclf__estimator__min_samples_split': [2, 3]
    # }

    # cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline
    


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate a multi-output classification model and print classification reports.

    This function takes a trained multi-output classification model, test data, and category names,
    and prints a classification report for each category.

    Args:
        model: A trained multi-output classification model.
        X_test (pandas.Series): The test data for model evaluation.
        y_test (pandas.DataFrame): The true labels for the test data.
        category_names (list): List of category names.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    # Loop through each column
    for i, col in enumerate(category_names):
        print(f"Classification Report for '{col}':")
        
        # Get the true labels for the current column
        true_labels = y_test[col] 
        
        # Get the predicted labels for the current column
        predicted_labels = y_pred[:, i]
        
        # Generate the classification report
        report = classification_report(true_labels, predicted_labels)
        print(report)
        print("-" * 60)  # Separator for clarity



def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file using pickle.

    Args:
        model: The trained machine learning model to be saved.
        model_filepath (str): The filepath where the model will be saved.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
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