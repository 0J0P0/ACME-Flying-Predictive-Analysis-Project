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

def extract_record(day, aircraft):
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

    record = []

    return record


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


def classifier_pipe(day: str, aircraft: str, model_name: str, model_path: str = './'):
    """
    ...
    """

    record = extract_record(day, aircraft)
    formatted_record = format_record(record)

    model = mlflow.spark.load_model(model_path + model_name)
    prediction = model.transform(formatted_record)

    print(f'Prediction for aircraft {aircraft} on day {day}: {prediction.select("prediction").first()[0]}')