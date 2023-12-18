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
from colorama import Fore
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.sql.functions import avg, lit, to_date, col, substring, expr, date_add

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
    
    matrix = sensor_data.join(
        kpis,
        (
            (sensor_data['aircraft id'] == kpis['aircraftid']) &
            (sensor_data['date'] == kpis['timeid'])
        ),
        'inner').drop('aircraftid', 'timeid').cache()
    
    # print(matrix.count())
    
    matrix = matrix.withColumn('date', col('date').cast('date'))
    labels = labels.withColumn('starttime', col('starttime').cast('date'))

    matrix = matrix.join(
        labels,
        (
            (col('aircraft id') == col('aircraftregistration')) &
            (col('starttime').between(col('date'), date_add(col('date'), 7)))
        ),
        'left').cache()
    
    # print(matrix.count())

    matrix = matrix.withColumn('label', expr('CASE WHEN aircraftregistration IS NOT NULL THEN 1 ELSE 0 END'))

    matrix = matrix.drop('aircraftregistration', 'starttime')

    # ...
    matrix = matrix.dropDuplicates(['aircraft id', 'date'])

    # print(matrix.count())
    
    # matrix.createOrReplaceTempView("matrix")
    # labels.createOrReplaceTempView("labels")

    # matrix = spark.sql("""
    #     SELECT
    #         matrix.*,
    #                    CASE WHEN EXISTS (
    #                         SELECT
    #                             COUNT(*)
    #                         FROM 
    #                             labels l2, matrix m2
    #                         WHERE l2.starttime BETWEEN m2.date AND DATE_ADD(m2.date, 7) AND l2.aircraftregistration = m2.`aircraft id`
    #                     ) THEN 1 ELSE 0 END AS label
    #     FROM
    #         matrix
    #     LEFT JOIN
    #         labels
    #     ON
    #         matrix.`aircraft id` = labels.aircraftregistration AND matrix.date = labels.starttime
    #     """)

    return format_columns(matrix)


def extract_labels(spark: SparkSession, amos_properties: dict, record: tuple = None) -> DataFrame:
    """
    Extracts the maintenance labels from the AMOS database and returns a DataFrame with the aircraft registration, the date and the label.

    Parameters
    ----------
    spark : SparkSession
        SparkSession object.
    amos_properties : dict
        Dictionary with the properties to connect to the AMOS database.

    Returns
    -------
    df : pyspark.sql.DataFrame
        DataFrame with the aircraft registration, the date and the label.
    """

    labels = spark.read.jdbc(url=amos_properties['url'],
                           table='oldinstance.operationinterruption',
                           properties=amos_properties)

    # print(labels.count())
    labels = labels.where(col('subsystem') == '3453')
    # print(labels.count())
    # labels = labels.withColumn('label', lit(1))
    labels = labels.withColumn('starttime', to_date(col('starttime'), 'yyyy-MM-dd'))
    # labels = labels.select('aircraftregistration', 'starttime', 'label')
    labels = labels.select('aircraftregistration', 'starttime')
    # labels = labels.groupBy('aircraftregistration', 'starttime').agg(lit(1).alias('label'))
    # print(labels.count())
    
    return labels


def extract_dw_data(spark: SparkSession, dbw_properties: dict, record: tuple = None) -> DataFrame:
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

    kpis = spark.read.jdbc(url=dbw_properties["url"],
                           table="public.aircraftutilization",
                           properties=dbw_properties)
    # print(kpis.count())
    # kpis = kpis.groupBy('aircraftid', 'timeid').agg(
    #     sum('flighthours').alias('flighthours'),
    #     sum('flightcycles').alias('flightcycles'),
    #     sum('delayedminutes').alias('delayedminutes'),
    # )
    # print(kpis.count()) 

    kpis = kpis.select('aircraftid', 'timeid', 'flighthours', 'flightcycles', 'delayedminutes')

    return kpis


def extract_sensor_data(spark: SparkSession, filepath: str, record: tuple = None) -> DataFrame:
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

    ####################################################
    sensors = df_set[list(df_set.keys())[0]]
    for i in range(1, len(df_set)):
        sensors = sensors.union(df_set[list(df_set.keys())[i]])

    sensors = sensors.groupBy("aircraft id", "date").agg(avg("value").alias("avg_sensor"))

    return sensors


def read_saved_matrix(spark: SparkSession) -> DataFrame:
    """
    .
    """

    matrix = spark.read.csv('./resources/matrix', header=True)
    return format_columns(matrix)


def managment_pipe(spark: SparkSession, dbw_properties: dict, amos_properties: dict, stage: str, filepath: str = './resources/trainingData/', save: bool = True, record: tuple = None) -> DataFrame:
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
    amos_properties : dict
        Dictionary with the properties to connect to the AMOS database.

    Returns
    -------
    matrix : pyspark.sql.DataFrame
        Matrix with the gathered data.
    """
    
    if os.path.exists('./resources/matrix') and record is None and stage in ['all', 'management']:
        matrix = read_saved_matrix(spark)
    else:
        print(f'{Fore.YELLOW}Extarcting sensor data...{Fore.RESET}')
        sensor_data = extract_sensor_data(spark, filepath, record).cache()
        # print(sensor_data.count())
        
        print(f'{Fore.YELLOW}Extarcting KPIs data...{Fore.RESET}')
        kpis = extract_dw_data(spark, dbw_properties, record).cache()
        # print(kpis.count())

        print(f'{Fore.YELLOW}Extarcting maintenance labels...{Fore.RESET}')
        labels = extract_labels(spark, amos_properties, record).cache()
        # print(labels.count())

        print(f'{Fore.YELLOW}Joining dataframes...{Fore.RESET}')
        matrix = join_dataframes(spark, sensor_data, kpis, labels)

        # print(1)
        if save:
            print(f'{Fore.YELLOW}Storing matrix...{Fore.RESET}')
            matrix.write.csv('./resources/matrix', header=True)
        # print(2)

        # matrix = spark.read.csv('./resources/matrix', header=True)
        # matrix = format_columns(matrix)

    return matrix