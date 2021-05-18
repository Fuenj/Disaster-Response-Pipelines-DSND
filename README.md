# Disaster-Response-Pipelines-DSND
Udacity Data Scientist Nanodegree Project

# Table of Contents

1. Project Motivation
2. Project details
3. Instructions
4. Results

# 1. Project Motivation
The aim of this project is to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. It include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. We analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The Project is divided in the following Sections:

#### 1. Data processing: 
This is a data cleaning pipeline that loads the messages and categories datasets, merges the two datasets, cleans the data and save it in a SQLite database.
#### 2. ML Pipeline
This is a machine learning pipeline that loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set and exports the final model as a pickle file.
#### 3. Web Application
This app aims to show final model results and visulaizations using Plotly.

# 2. Project details
The following are the main files available in this repository:

` ETL Pipeline Preparation.ipynb`  - a notebook which contains the first part of the data pipeline namely,Extract, Transform, and Load process. Here, you will read the dataset, clean the data with pandas, and then store it in a SQLite database. 

`process_data.py` - This file include the cleaning code in the ETL script.

` ML Pipeline Preparation.ipynb` - a notebook which contains the machine learning portion, namely, split the data into a training set and a test set, Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, it export the model to a pickle file. 

`train_classifier.py` - Include the final machine learning code.

`run.py` - Flask file that runs app

`template files`- Templates for these scripts are provided in this file, namely, the main page of the web app and the classification result page.

Here's the file structure of the project:

- app
  - template
     * master.html  # main page of web app
     * go.html  # classification result page of web app

  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py  
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

- README.md

# 3. Instructions
1)- Run the following commands to set up your database and model:

   - To run ETL pipeline that cleans data and stores in database: 
  
 `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 
   - To run ML pipeline that trains classifier:
   
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2)- Run the following command in the app's directory to run your web app:

 `python app/run.py`

3)- Go to http://0.0.0.0:3001/

# 4. Results
This section shows the main results in the application. Below are a few screenshots of the web app:

![image](https://user-images.githubusercontent.com/73600826/118540569-8d672a00-b705-11eb-88af-80b07eb9f520.png)

![image](https://user-images.githubusercontent.com/73600826/118324587-18061a00-b4b7-11eb-9137-3dbbeb0c4b00.png)



