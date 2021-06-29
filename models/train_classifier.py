import sys
import nltk
import re
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
            Functions:
                load data from sqlite file then extract message and category data to train and test
            Arguments:
                database_filepath: path to database
            Output:
                x: message data
                y: category data
                category_names : category column name
    """
    sqlite_path = "sqlite:///{0}".format(database_filepath)
    engine = create_engine(sqlite_path)
    df = pd.read_sql_table('message', engine)
    x = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return x, y, category_names


def tokenize(text):
    """
            Functions:
                tokenize message
            Arguments:
                text: input text
            Output:
                clean_tokens: message data that has been tokenize
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
            Functions:
                create model pipeline and grid search then return a optimize model
            Output:
                model: model for use
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        # 'clf__estimator__n_estimators': [50, 100],
        # 'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
        # 'vect__max_df': (0.75, 1.0),
        # 'clf__estimator__n_estimators': [10, 20],
        # 'clf__estimator__min_samples_split': [2, 5]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, x_test, y_test, category_names):
    """
            Functions:
                evaluate performance of the model
    """
    predict = model.predict(x_test)
    for col in range(predict.shape[1]):
        print('Category: ', category_names[col])
        print(classification_report(y_test.values[:, col], predict[:, col]), '====================================')


def save_model(model, model_filepath):
    """
            Functions:
                save to pkl file (Pickle)
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
            Functions:
                run the machine learning pipeline
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(x_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
