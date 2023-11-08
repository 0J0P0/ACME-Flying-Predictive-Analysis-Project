"""
This pipeline generates a matrix where the rows denote the information of an aircraft per day, and the columns refer to the FH, FC and DM KPIs, and the average measurement of the 3453 sensor. Thus, this pipeline must:

    - Read the sensor measurements (extracted from the CSV files) related to a certain aircraft A and average it per day.
    - Once you have the average measurement of the sensor per day, enrich it with the KPIs related to A from the Data Warehouse (at the same granularity level).
    - Importantly, since we are going to use supervised learning (see the Data Analysis Pipeline below), we need to label each row with a label: either unscheduled maintenance or no maintenance predicted in the next 7 days for that flight.
    - Generate a matrix with the gathered data and store it.
"""

import os
from colorama import Fore
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

def extract_dw_data(spark: SparkSession, db_properties: dict):
    """

    """

    data = spark.read.jdbc(url=db_properties["url"],
                           table="oldinstance.operationinterruption",
                           properties=db_properties)

    df = data.select("flightid","airport","duration")
    df.show()

    return df


def extract_sensor_data(filepath: str, spark: SparkSession):
    """

    """

    sensor_data = {}
    files = os.listdir(filepath)

    for filename in files:
        if filename.endswith(".csv"):
            flight = filename.split('-')
            aircraft_day = flight[4] + '-' + flight[5][:3] + '_' + flight[0]

            df = spark.read.csv(filepath + filename, sep=';')
    
            if aircraft_day not in sensor_data:
                sensor_data[aircraft_day] = df
            else:
                sensor_data[aircraft_day] = sensor_data[aircraft_day].union(df)
    
    rows = []
    for aircraft_day, df in sensor_data.items():
        avg_sensor = df.select(avg(df['_c2'])).collect()[0][0]
        aircraft, day = aircraft_day.split('_')
        day = '20' + day[:2] + '-' + day[2:4] + '-' + day[4:]
        
        rows.append((aircraft, day, avg_sensor))

    columns = ['aircraft id', 'day', 'avg_sensor']
    sensor_data_df = spark.createDataFrame(rows, columns)

    return sensor_data_df


def managment_pipe(filepath: str, spark: SparkSession, db_properties: dict):
    """

    """

    print('-'*50 + '\n' + f'{Fore.YELLOW}Extracting sensor data...{Fore.RESET}')
    # sensor_data_df = extract_sensor_data(filepath, spark)
    print(f'{Fore.YELLOW}Extarcting KPIs data...{Fore.RESET}')
    kpis = extract_dw_data(spark, db_properties)

    return kpis
