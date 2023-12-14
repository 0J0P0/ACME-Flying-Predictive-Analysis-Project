##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Classifier Pipeline                                            #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################


"""
This pipepline recieves an aircraft and a day, replicates the process of the analysis pipleine to generate a 
record and then uses the best model to classify the record in maintenance or no maintenance.

The steps are the following:
- Recieve an aircraft and a day.
- Generate a record with that aircraft and day.
- Format the record so it can be used by the model.
- Load the best model.
- Make a prdiction with the record and the model.
"""

##############################################################################################################
#                                                                                                            #
# Imports                                                                                                    #
#                                                                                                            #
##############################################################################################################

import mlflow
import pandas as pd
from colorama import Fore
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################


def format_record(record):
    """
    Formats the record so it can be used by the model.

    Parameters
    ----------
    record : pandas.DataFrame
        Record to format.

    Returns
    -------
    record : pyspark.sql.DataFrame
        Formatted record.
    """

    aircraftIndexer = StringIndexer(inputCol='aircraft id', outputCol='aircraft_id')
    dateIndexer = StringIndexer(inputCol='date', outputCol='date_id')
    assembler = VectorAssembler(inputCols=['aircraft_id', 'date_id', 'avg_sensor', 'flighthours', 
                                           'flightcycles', 'delayedminutes'],
                                outputCol='features')

    pipeline = Pipeline(stages=[aircraftIndexer, dateIndexer, assembler])
    record = pipeline.fit(record).transform(record)
    
    return record.select('features')


def extract_record(day: str, aircraft: str, matrix: DataFrame):
    """
    Extracts the record for the given day and aircraft.

    Parameters
    ----------
    day : str
        Day to extract the record.
    aircraft : str
        Aircraft to extract the record.

    Returns
    -------
    record : pandas.DataFrame
        Record for the given day and aircraft.
    """

    record = matrix.filter(matrix['date'] == day).filter(matrix['aircraft id'] == aircraft)
    record = record.select('aircraft id', 'date', 'avg_sensor', 'flighthours', 'flightcycles', 'delayedminutes')

    return record


def valid_input(day: str, aircraft: str):
    """
    Checks if the input is valid. The input is valid if it has the following format:
    - day: YYYY-MM-DD
    - aircraft: XX-XXX

    Parameters
    ----------
    day : str
        Day to check.
    aircraft : str
        Aircraft to check.

    Returns
    -------
    valid : bool
        True if the input is valid, False otherwise.
    """

    aux = day.split('-')
    valid = len(aux) == 3 and len(aux[0]) == 4 and len(aux[1]) == 2 and len(aux[2]) == 2
    
    aux = aircraft.split('-')
    valid = valid and len(aux) == 2 and len(aux[0]) == 2 and len(aux[1]) == 3

    if not valid:
        print(f'{Fore.RED}Invalid input.{Fore.RESET}')

    return valid


def classifier_pipe(model_name: str, matrix: DataFrame, model_path: str = './models/'):
    """
    ...
    """

    day = input('Enter a day (YYYY-MM-DD): ')
    aircraft = input('Enter an aircraft (XX-XXX): ')

    while valid_input(day, aircraft):

        record = extract_record(day, aircraft, matrix)
        formatted_record = format_record(record)

        model = mlflow.spark.load_model(model_path + model_name)
        prediction = model.transform(formatted_record)

        print(f'Prediction for aircraft {aircraft} on day {day}: {prediction.select("prediction").first()[0]}')

        day = input('Enter a day (YYYY-MM-DD): ')
        aircraft = input('Enter an aircraft (XX-XXX): ')