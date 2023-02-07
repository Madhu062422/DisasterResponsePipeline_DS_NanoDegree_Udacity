# Udacity_DS_Nanodegree_Disaster_Response_Pipeline

## Project Description:

In this project, we will be analyzing disaster data from Appen (formally Figure 8)to build model for an API that can classify realtime disaster messages. The project consists of building 

    a. An ETL Pipeline : That extracts data from source, cleans and write the processed data into SQLite DB
    b. Machine Learning Pipeline : to categorize the messages to send them to an appropriate disaster relief agency.
    c. Web Application : where an emergency worker can input a new message and get classification results in several categories

Since the model could classify a message into more than one category, this is a multi-label classification task. The model will be trained using the datset from Figure Eight that contains real messages which were sent during disaster events. 

## Web App
<img width="1434" alt="Screenshot 2023-01-30 at 10 26 30 AM" src="https://user-images.githubusercontent.com/64095099/216180855-5b17eb19-205a-4720-8669-d7ac1cf8a12e.png">

<img width="1186" alt="Screenshot 2023-02-06 at 10 51 41 PM" src="https://user-images.githubusercontent.com/64095099/217114321-0863e129-9325-4735-9216-7cc10df5c277.png">


## File Description
<img width="600" alt="Screenshot 2023-02-02 at 1 43 57 PM" src="https://user-images.githubusercontent.com/64095099/216341358-00a6b164-f43b-4c5a-b55a-bde025aa71e9.png">


## Dependencies 
Python 3
NumPy, SciPy, Pandas, Sciki-Learn
NLTK (for Natural Language Processing)
SQLalchemy (SQLlite Database)
Pickle (Model loading and storing)
Flask, Plotly (Data Visualization and Web Application)

## Instructions 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements

Many thanks to Figure-8 for making this available to Udacity for training purposes. 
Special thanks to udacity for the training. 

## Git hub Link : https://github.com/Madhu062422/DisasterResponsePipeline_DS_NanoDegree_Udacity/tree/master

