# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#Instruction)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The necessary libraries are listed in 'requirements.txt' file.
The code is using Python versions 3.\*.

## Project Overview<a name="overview"></a>

Following the event of disasters, there's a flood of millions of messages millions of messages, making it challenging for response organizations to identify the most critical ones. Different organizations address specific disaster aspects like water or blocked roads. The purpose of this project is to build a model that classifies disaster messages so that messages are sent to corresponding organizations ((E.g. messages related to injuries should be sent to medical staff).

This project uses real messages that were sent during disaster events from Appen (formally Figure 8) to build that model. The model is then made available as an API and deployed into a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions <a name="files"></a>

Here's the file structure of the project:

- app
  | - templates
  | |- master.html # main page of web app
  | |- go.html # classification result page of web app
  |- run.py # Flask file that runs app

- data
  |- disaster_categories.csv # data to process
  |- disaster_messages.csv # data to process
  |- process_data.py
  |- InsertDatabaseName.db # database to save clean data to

- models
  |- train_classifier.py
  |- classifier.pkl # saved model

- README.md
- requirements.txt
- .gitignore

## Project Components <a name="results"></a>

- ETL pipeline:
  'process_data.py' will read the dataset, clean the data, and then store it in a SQLite database.
- ML pipeline:
  'train_classifier.py' creates a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, the model is exported to a pickle file.
- Flask App:
  The flask app displays the results. Users can input a message to get its categorization.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Appen (formally Figure 8) for the data.
