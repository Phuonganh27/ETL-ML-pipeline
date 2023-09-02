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

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='CleanData', con=engine)
    X = df['message']
    y = df.drop(columns=['id','message', 'original', 'genre'])
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Tokenize and preprocess input text data.

    Args:
        text (str): Input text to tokenize.

    Returns:
        list: List of preprocessed tokens.
    """
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def build_model():
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