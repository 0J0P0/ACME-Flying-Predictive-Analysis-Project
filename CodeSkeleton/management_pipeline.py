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
from colorama import Fore
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import avg, sum, lit, date_format, to_date, substring


##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################


def join_dataframes(spark: SparkSession, sensor_data, kpis, labels):
    """
    l
    """

    matrix = sensor_data.join(kpis, (sensor_data['aircraft id'] == kpis['aircraftid']) & (sensor_data['day'] == kpis['timeid']), 'inner').drop('aircraftid', 'timeid')

    # make a lef join of matrix with labels, conserving all the rows of labels
    matrix2 = matrix.join(labels, (matrix['aircraft id'] == labels['aircraftregistration']) & (matrix['day'] == labels['date']), how='left').drop('aircraftregistration', 'date')

    newSchema = StructType([
        StructField("aircraft id", StringType(), True),
        StructField("day", StringType(), True),
        StructField("avg_sensor", DoubleType(), True),
        StructField("flighthours", DoubleType(), True),
        StructField("flightcycles", IntegerType(), True),
        StructField("delayedminutes", DoubleType(), True),
        StructField("kind", StringType(), True)
    ])

    matrix2 = matrix2.withColumn('avg_sensor', matrix2['avg_sensor'].cast(DoubleType())) \
                        .withColumn('flighthours', matrix2['flighthours'].cast(DoubleType())) \
                        .withColumn('flightcycles', matrix2['flightcycles'].cast(IntegerType())) \
                        .withColumn('delayedminutes', matrix2['delayedminutes'].cast(DoubleType()))
    
    matrix2 = spark.createDataFrame(data=matrix2.rdd, schema=newSchema, verifySchema=True)

    return matrix.orderBy('aircraft id', 'day'), matrix2.orderBy('aircraft id', 'day')


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

    data = spark.read.jdbc(url=damos_properties["url"],
                           table="oldinstance.operationinterruption",
                           properties=damos_properties)

    df = data.withColumn('date',
                         date_format(to_date(substring('flightid', 1, 6), 'ddMMyy'),'yyyy-MM-dd'))

    df = df.groupBy('date', 'aircraftregistration').agg(lit('Maintenance').alias('kind'))

    return df.orderBy('aircraftregistration', 'date')


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
    ).orderBy('aircraftid', 'timeid')

    return df


def extract_sensor_data(filepath: str, spark: SparkSession):
    """
    Extracts the sensor data from the csv files and returns a DataFrame with the average measurement of the 3453 sensor per day.

    Parameters
    ----------
    filepath : str
        Path to the folder where the csv files are stored.
    spark : SparkSession
        SparkSession object.

    Returns
    -------
    sensor_data_df : pyspark.sql.DataFrame
        DataFrame with the average measurement of the 3453 sensor per day.
    """
    
    rows = []
    files = os.listdir(filepath)
    
    for filename in files:
        if filename.endswith(".csv"):
            flight = filename.split('-')
            aircraft, day = flight[4] + '-' + flight[5][:3], flight[0]
            day = '20' + day[4:] + '-' + day[2:4] + '-' + day[:2]

            df = spark.read.csv(filepath + filename, sep=';')
            avg_sensor = df.select(avg(df['_c2']).alias('avg_sensor')).collect()[0]['avg_sensor']
            
            rows.append((aircraft, day, avg_sensor))

    columns = ['aircraft id', 'day', 'avg_sensor']
    sensor_data_df = spark.createDataFrame(rows, columns)
    
    # Grouping by aircraft id and day, selecting the average sensor value
    sensor_data_df = sensor_data_df.groupBy('aircraft id', 'day').agg({'avg_sensor': 'avg'})
    sensor_data_df = sensor_data_df.withColumnRenamed('avg(avg_sensor)', 'avg_sensor')

    return sensor_data_df.orderBy('aircraft id', 'day')


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

    matrix, matrix2 = join_dataframes(spark, sensor_data, kpis, labels)
    print(f'{Fore.GREEN}End of the Managment Pipeline{Fore.RESET}' + '\n' + '-'*50)

    return labels, matrix2