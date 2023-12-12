##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Managment Pipeline                                             #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################


"""
This pipeline generates a matrix where the rows denote the information of an aircraft per day, and the columns refer to the FH, FC and DM KPIs, and the average measurement of the 3453 sensor. Thus, this pipeline must:

    - Read the sensor measurements (extracted from the CSV files) related to a certain aircraft A and average it per day.
    - Once you have the average measurement of the sensor per day, enrich it with the KPIs related to A from the Data Warehouse (at the same granularity level).
    - Importantly, since we are going to use supervised learning (see the Data Analysis Pipeline below), we need to label each row with a label: either unscheduled maintenance or no maintenance predicted in the next 7 days for that flight.
    - Generate a matrix with the gathered data and store it.
"""

##############################################################################################################
#                                                                                                            #
# Imports                                                                                                    #
#                                                                                                            #
##############################################################################################################

import os
import pandas as pd
from colorama import Fore
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, DoubleType, IntegerType, StructType, StructField
from pyspark.sql.functions import avg, sum, lit, to_date, col, substring

##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################

def format_columns(matrix: DataFrame) -> DataFrame:
    """
    Formats the matrix so that the columns are the FH, FC and DM KPIs, and the average measurement of the 3453 sensor, and the label.

    Parameters
    ----------
    matrix : pyspark.sql.DataFrame
        Matrix with the gathered data.

    Returns
    -------
    matrix : pyspark.sql.DataFrame
        Matrix with the gathered data.
    """

    matrix = matrix.withColumn('aircraft id', matrix['aircraft id'].cast(StringType())) \
        .withColumn('date', matrix['date'].cast(StringType())) \
        .withColumn('avg_sensor', matrix['avg_sensor'].cast(DoubleType())) \
        .withColumn('flighthours', matrix['flighthours'].cast(DoubleType())) \
        .withColumn('flightcycles', matrix['flightcycles'].cast(IntegerType())) \
        .withColumn('delayedminutes', matrix['delayedminutes'].cast(IntegerType())) \
        .withColumn('label', matrix['label'].cast(IntegerType()))

    return matrix


def join_dataframes(spark: SparkSession, sensor_data: DataFrame, kpis: DataFrame, labels: DataFrame) -> DataFrame:
    """
    Joins the sensor measurements, the KPIs and the labels, and returns a DataFrame with the average measurement per flight per day, the FH, FC and DM KPIs, and the label.

    Parameters
    ----------
    spark : SparkSession
        SparkSession object.
    sensor_data : pyspark.sql.DataFrame
        DataFrame with the average measurement per flight per day.
    kpis : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs.
    labels : pyspark.sql.DataFrame
        DataFrame with the aircraft registration, the date and the label.

    Returns
    -------
    matrix : pyspark.sql.DataFrame
        DataFrame with the average measurement per flight per day, the FH, FC and DM KPIs, and the label.
    """
    
    matrix = sensor_data.join(kpis, (sensor_data['aircraft id'] == kpis['aircraftid']) & (sensor_data['date'] == kpis['timeid']), 'inner').drop('aircraftid', 'timeid')

    matrix = matrix.join(labels, (matrix['aircraft id'] == labels['aircraftregistration']) & (matrix['date'] == labels['starttime']), 'left').drop('aircraftregistration', 'starttime')#.cache()

    matrix = matrix.fillna(0, subset=['label'])

    # matrix = matrix.toPandas()
    # labels = labels.toPandas()

    # matrix['date'] = pd.to_datetime(matrix['date'])
    # labels['starttime'] = pd.to_datetime(labels['starttime'])

    # for i, row in matrix.iterrows():
    #     if row['label'] == None:
    #         seven_day_label = labels[(labels['aircraftregistration'] == row['aircraft id']) & (labels['starttime'] > row['date']) & (labels['starttime'] <= row['date'] + pd.DateOffset(days=7))]

    #         if seven_day_label.empty:
    #             matrix.at[i, 'label'] = 0
    #         else:
    #             matrix.at[i, 'label'] = 1

    # matrix = spark.createDataFrame(data=matrix, verifySchema=True)

    return format_columns(matrix)


def extract_labels(spark: SparkSession, damos_properties: dict) -> DataFrame:
    """
    Extracts the maintenance labels from the AMOS database and returns a DataFrame with the aircraft registration, the date and the label.

    Parameters
    ----------
    spark : SparkSession
        SparkSession object.
    damos_properties : dict
        Dictionary with the properties to connect to the AMOS database.

    Returns
    -------
    df : pyspark.sql.DataFrame
        DataFrame with the aircraft registration, the date and the label.
    """

    labels = spark.read.jdbc(url=damos_properties['url'],
                           table='oldinstance.operationinterruption',
                           properties=damos_properties)

    labels = labels.withColumn('starttime', to_date(col('starttime'), 'yyyy-MM-dd'))
    labels = labels.groupBy('aircraftregistration', 'starttime').agg(lit(1).alias('label'))
    
    return labels


def extract_dw_data(spark: SparkSession, dbw_properties: dict) -> DataFrame:
    """
    Extracts the KPIs related to an aircraft from the Data Warehouse and returns a DataFrame with the FH, FC and DM KPIs.

    Parameters
    ----------
    spark : SparkSession
        SparkSession object.
    dbw_properties : dict
        Dictionary with the properties to connect to the Data Warehouse.

    Returns
    -------
    df : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs.
    """

    data = spark.read.jdbc(url=dbw_properties["url"],
                           table="public.aircraftutilization",
                           properties=dbw_properties)

    df = data.groupBy('aircraftid', 'timeid').agg(
        sum('flighthours').alias('flighthours'),
        sum('flightcycles').alias('flightcycles'),
        sum('delayedminutes').alias('delayedminutes'),
    )

    return df


def extract_sensor_data(filepath: str, spark: SparkSession) -> DataFrame:
    """
    Extracts the sensor measurements from the CSV files and returns a DataFrame with the average measurement per flight per day.

    Parameters
    ----------
    filepath : str
        Path to the folder where the csv files are stored.
    spark : SparkSession
        SparkSession object.

    Returns
    -------
    sensors : pyspark.sql.DataFrame
        DataFrame with the average measurement per flight per day.
    """

    def extract_aircraft_id(file_name):
        flightid = file_name.split('-')
        return flightid[4] + '-' + flightid[5].split('.')[0]

    df_set = dict()
    all_files = os.listdir(filepath)

    for file in all_files:
        if file.endswith('.csv'):
            aircraft_id = extract_aircraft_id(file)
            
            df = spark.read.option('header', 'true').option('delimiter', ';').csv(filepath + file)
            df = df.withColumn('date', substring('date', 1, 10))
            df = df.withColumn('aircraft id', lit(aircraft_id))
            df = df.select('aircraft id', 'date', 'value')
            
            if aircraft_id in df_set:
                df_set[aircraft_id] = df_set[aircraft_id].union(df)
            else:
                df_set[aircraft_id] = df

    sensors = df_set[list(df_set.keys())[0]]
    for i in range(1, len(df_set)):
        sensors = sensors.union(df_set[list(df_set.keys())[i]])


    sensors = sensors.groupBy("aircraft id", "date").agg(avg("value").alias("avg_sensor"))

    return sensors


def managment_pipe(filepath: str, spark: SparkSession, dbw_properties: dict, damos_properties: dict) -> DataFrame:
    """
    Managment Pipeline. This pipeline generates a matrix where the rows denote the information of an aircraft per day, and the columns refer to the FH, FC and DM KPIs, and the average measurement of the 3453 sensor. 

    Parameters
    ----------
    filepath : str
        Path to the folder where the csv files are stored.
    spark : SparkSession
        SparkSession object.
    dbw_properties : dict
        Dictionary with the properties to connect to the Data Warehouse.
    damos_properties : dict
        Dictionary with the properties to connect to the AMOS database.

    Returns
    -------
    matrix : pyspark.sql.DataFrame
        Matrix with the gathered data.
    """
    
    if os.path.exists('./resources/matrix'):
        matrix = spark.read.csv('./resources/matrix', header=True)
        matrix = format_columns(matrix)
    else:
        print(f'{Fore.YELLOW}Extarcting sensor data...{Fore.RESET}')
        sensor_data = extract_sensor_data(filepath, spark)
        # print(sensor_data.count())
        
        print(f'{Fore.YELLOW}Extarcting KPIs data...{Fore.RESET}')
        kpis = extract_dw_data(spark, dbw_properties)
        # print(kpis.count())

        print(f'{Fore.YELLOW}Extarcting maintenance labels...{Fore.RESET}')
        labels = extract_labels(spark, damos_properties)
        # print(labels.count())

        matrix = join_dataframes(spark, sensor_data, kpis, labels)

        # print(1)
        matrix.write.csv('./resources/matrix', header=True)
        # print(2)

        matrix = spark.read.csv('./resources/matrix', header=True)
        matrix = format_columns(matrix)

    return matrix