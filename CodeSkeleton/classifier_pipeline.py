##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Classifier Pipeline                                            #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################


"""
This pipepline recieves multiple inputs of the form (aircraft, day) and makes a maintenance prediction. First it validates the input, then it completes the record with the sensor and KPIs data, and finally it makes a prediction with the selected model. 

The steps are the following:
- Recieve an aircraft and a day.
- Validate the input.
- Complete the record with the sensor and KPIs data.
- Make a prediction with the selected model.

If the input is not valid, the pipeline will ask for a new input. If the record is not complete, the pipeline will show an error message and ask for a new input.
"""

##############################################################################################################
#                                                                                                            #
# Imports                                                                                                    #
#                                                                                                            #
##############################################################################################################

import mlflow
from colorama import Fore
from pyspark.ml import Pipeline 
import management_pipeline as mp
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler

##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################


def format_record(record: DataFrame):
    """
    Formats the record so it can be used by the model.

    Parameters
    ----------
    record : pyspark.sql.DataFrame
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


def extract_record(spark: SparkSession, day: str, aircraft: str, dbw_properties: dict):
    """
    Extracts the record from the available data.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Spark session.
    day : str
        Day to extract.
    aircraft : str
        Aircraft to extract.
    dbw_properties : dict
        Dictionary with the properties to connect to the database.

    Returns
    -------
    record : pyspark.sql.DataFrame
        Record extracted.
    """

    print(f'{Fore.YELLOW}Completing record with sensor data...{Fore.RESET}')
    sensor_data = mp.extract_sensor_data(spark=spark, record=(aircraft, day))
    
    if sensor_data is None or sensor_data.count() == 0:
        print(f'{Fore.RED}No sensor data for aircraft {aircraft} on day {day}.{Fore.RESET}')
        return None, True

    print(f'{Fore.YELLOW}Completing record with KPIs data...{Fore.RESET}')
    kpis = mp.extract_dw_data(spark, dbw_properties, (aircraft, day))

    if kpis.count() == 0:
        print(f'{Fore.RED}No KPIs data for aircraft {aircraft} on day {day}.{Fore.RESET}')
        return None, True

    print(f'{Fore.YELLOW}Joining information...{Fore.RESET}')
    record = sensor_data.join(
        kpis,
        (
            (sensor_data['aircraft id'] == kpis['aircraftid']) &
            (sensor_data['date'] == kpis['timeid'])
        ),
        'inner').drop('aircraftid', 'timeid')
    
    if record.count() == 0:
        print(f'{Fore.RED}No common data from sensor and KPIs for aircraft {aircraft} on day {day}.{Fore.RESET}')
        return None, True

    return format_record(record), False


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


def classifier_pipe(spark: SparkSession, model, dbw_properties: dict):
    """
    Run-time classifier pipeline. Recieves multiple inputs of the form (aircraft, day) and makes if the aircraft is going for unscheduled maintenance or not.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Spark session.
    model : pyspark.ml.PipelineModel
        Model to use.
    dbw_properties : dict
        Dictionary with the properties to connect to the database.

    Returns
    -------
    None    
    """

    day = input('Enter a day (YYYY-MM-DD) or "exit" to end: ')
    aircraft = input('Enter an aircraft (XX-XXX) or "exit" to end: ')

    while day != 'exit' and aircraft != 'exit':
        if valid_input(day, aircraft):
            try:
                record, empty = extract_record(spark, day, aircraft, dbw_properties)

                if not empty:
                    prediction = model.transform(record)
                    pred = prediction.select("prediction").first()[0]
                    if pred == 0.0:
                        pred = 'No maintenance'
                    else:
                        pred = 'Maintenance'
                    print(f'{Fore.MAGENTA}Prediction for aircraft {aircraft} on day {day}: {pred}{Fore.RESET}')
            except Exception as e:
                print(f'{Fore.RED}Error in aircraft {aircraft} on day {day}. {e}{Fore.RESET}')

        day = input('Enter a day (YYYY-MM-DD): ')
        aircraft = input('Enter an aircraft (XX-XXX): ')