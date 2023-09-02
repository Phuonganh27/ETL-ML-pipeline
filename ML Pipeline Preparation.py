#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
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


# In[2]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[3]:


# load data from database
engine = create_engine('sqlite:///CleanData.db')
df = pd.read_sql_table(table_name='CleanData', con=engine)
X = df['message']
y = df.drop(columns=['id','message', 'original', 'genre'])


# In[55]:


X.shape


# In[56]:


y.shape


# ### 2. Write a tokenization function to process your text data

# In[5]:


# def tokenize(text):
#     tokens = word_tokenize(text.lower())
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens ]

#     return clean_tokens

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

# test = X.apply(tokenize)


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tf', TfidfTransformer()),
    ('mclf', MultiOutputClassifier(RandomForestClassifier()))
])


X_train, X_test, y_train, y_test = train_test_split(X, y)


model = pipeline.fit(X_train, y_train)

# from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
(y_pred == y_test).mean()


# In[10]:


from sklearn.metrics import classification_report

column_names = y_test.columns

# Loop through each column
for i, col in enumerate(column_names):
    print(f"Classification Report for '{col}':")
    
    # Get the true labels for the current column
    true_labels = y_test[col] 
    
    # Get the predicted labels for the current column
    predicted_labels = y_pred[:, i]
    
    # Generate the classification report
    report = classification_report(true_labels, predicted_labels)
    print(report)
    print("-" * 60)  # Separator for clarity


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[13]:


parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'mclf__estimator__n_estimators': [50, 100],
        'mclf__estimator__min_samples_split': [2, 3]
    }

cv = GridSearchCV(pipeline, param_grid=parameters)


# In[ ]:


model.get_params().keys()


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


new_model = cv.fit(X_train, y_train)
y_pred = new_model.predict(X_test)
labels = np.unique(y_pred)
confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
accuracy = (y_pred == y_test).mean()

print("Labels:", labels)
print("Confusion Matrix:\n", confusion_mat)
print("Accuracy:", accuracy)
print("\nBest Parameters:", cv.best_params_)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:


import pickle

# Assuming you have a trained scikit-learn model called 'model'
# Replace 'model.pkl' with the desired file name
with open('model.pkl', 'wb') as file:
    pickle.dump(new_model, file)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




