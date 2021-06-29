# Disaster Response Pipeline Project

### Project Description:

This project goal is to analyze emergency message that send during the disaster to predict that what massage category is that for using in help assistance. 
This project include 3 parts ETL process that load raw data and clean then load to sqlite database and then machine learning pipeline to create text classification to classify category of the message.

### Installation:
1. run virtual env or python env you want 
2. run requirements.txt

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py your_root_path_to_folder/models/DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
