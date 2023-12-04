##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project                                                                  #
#                                                                                                            #
# Input:        - Path to the folder where the csv files are stored.                                         #
#               - SparkSession object.                                                                       #
#               - Dictionary with the properties to connect to the Data Warehouse.                           #
#               - Dictionary with the properties to connect to the AMOS database.                            #
#                                                                                                            #
# Output:       - Matrix with the gathered data.                                                             #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
#                                                                                                            #
# Requirements: - pyspark.sql                                                                                #
#               - pyspark.sql.types                                                                          #
#               - pyspark.sql.functions                                                                      #
#               - os                                                                                         #
#               - colorama                                                                                   #
#                                                                                                            #
#############################################################################################################


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
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import avg, sum, lit, to_date, col, substring


##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################


def join_dataframes(spark: SparkSession, sensor_data: DataFrame, kpis: DataFrame, labels: DataFrame):
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

    matrix = matrix.join(labels, (matrix['aircraft id'] == labels['aircraftregistration']) & (matrix['date'] == labels['starttime']), how='left').drop('aircraftregistration', 'starttime')

    # matrix.fillna('No Maintenance', subset=['label'])

    matrix = matrix.toPandas()
    labels = labels.toPandas()

    matrix['date'] = pd.to_datetime(matrix['date'])
    labels['starttime'] = pd.to_datetime(labels['starttime'])

    for i, row in matrix.iterrows():
        if row['label'] == None:
            seven_day_label = labels[(labels['aircraftregistration'] == row['aircraft id']) & (labels['starttime'] > row['date']) & (labels['starttime'] <= row['date'] + pd.DateOffset(days=7))]

            if seven_day_label.empty:
                matrix.at[i, 'label'] = 'No Maintenance'
            else:
                matrix.at[i, 'label'] = 'Maintenance'

    matrix = spark.createDataFrame(data=matrix, verifySchema=True)

    newSchema = StructType([
        StructField("aircraft id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("avg_sensor", DoubleType(), True),
        StructField("flighthours", DoubleType(), True),
        StructField("flightcycles", IntegerType(), True),
        StructField("delayedminutes", IntegerType(), True),
        StructField("label", StringType(), True)
    ])

    matrix = matrix.withColumn('avg_sensor', matrix['avg_sensor'].cast(DoubleType())) \
                        .withColumn('flighthours', matrix['flighthours'].cast(DoubleType())) \
                        .withColumn('flightcycles', matrix['flightcycles'].cast(IntegerType())) \
                        .withColumn('delayedminutes', matrix['delayedminutes'].cast(IntegerType()))
    
    matrix = spark.createDataFrame(data=matrix.rdd, schema=newSchema, verifySchema=True)

    return matrix.orderBy('aircraft id', 'date')


def extract_labels(spark: SparkSession, damos_properties: dict):
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

    labels = spark.read.jdbc(url=damos_properties["url"],
                           table="oldinstance.operationinterruption",
                           properties=damos_properties)

    # que es mejor, hacer el select aqui o dejar que lo haga el group by?
    # labels = labels.select('aircraftregistration', 'starttime').distinct()

    labels = labels.withColumn("starttime", to_date(col("starttime"), 'yyyy-MM-dd'))

    labels = labels.groupBy('aircraftregistration', 'starttime').agg(lit('Maintenance').alias('label'))
    
    # return labels.orderBy('aircraftregistration', 'date')
    return labels


def extract_dw_data(spark: SparkSession, dbw_properties: dict):
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
    )#.orderBy('aircraftid', 'timeid')

    return df


def extract_sensor_data(filepath: str, spark: SparkSession):
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

    df_list = []
    all_files = os.listdir(filepath)
    for file in all_files:
        if file.endswith('.csv'):
            df = spark.read.option("header", "true").option("delimiter", ";").csv(filepath + file)
            
            aircraft_id = extract_aircraft_id(file)
            
            df = df.withColumn("aircraft id", lit(aircraft_id))
            df = df.withColumn("date", substring("date", 1, 10))
            
            df_list.append(df)

    sensors = df_list[0]
    for i in range(1, len(df_list)):
        sensors = sensors.union(df_list[i])

    sensors = sensors.groupBy("aircraft id", "date").agg(avg("value").alias("avg_sensor"))

    return sensors


def managment_pipe(filepath: str, spark: SparkSession, dbw_properties: dict, damos_properties: dict):
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

    print('-'*50 + '\n' + f'{Fore.CYAN}Start of the Managment Pipeline{Fore.RESET}')

    # suponemos que las fechas de los ficheros csv son las de los vuelos
    print(f'{Fore.YELLOW}Extarcting sensor data...{Fore.RESET}')
    sensor_data = extract_sensor_data(filepath, spark)
    # print(sensor_data.count())
    
    print(f'{Fore.YELLOW}Extarcting KPIs data...{Fore.RESET}')
    kpis = extract_dw_data(spark, dbw_properties)
    # print(kpis.count())

    # los labels son mirando a 7 dias vista para cada vuelo. Es decir que un vuelo tiene label maintenance si es que en los 7 dias siguientes tiene un vuelo con label maintenance
    print(f'{Fore.YELLOW}Extarcting maintenance labels...{Fore.RESET}')
    labels = extract_labels(spark, damos_properties)
    # print(labels.count())

    matrix = join_dataframes(spark, sensor_data, kpis, labels)
    print(f'{Fore.GREEN}End of the Managment Pipeline{Fore.RESET}' + '\n' + '-'*50)

    return matrix