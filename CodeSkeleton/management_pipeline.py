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
from pyspark.sql.functions import avg, sum, lit, to_date, col, substring


##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################


def join_dataframes(spark: SparkSession, sensor_data, kpis, labels):
    """
    l
    """

    matrix = sensor_data.join(kpis, (sensor_data['aircraft id'] == kpis['aircraftid']) & (sensor_data['date'] == kpis['timeid']), 'inner').drop('aircraftid', 'timeid')

    # make a lef join of matrix with labels, conserving all the rows of labels
    matrix2 = matrix.join(labels, (matrix['aircraft id'] == labels['aircraftregistration']) & (matrix['date'] == labels['starttime']), how='left').drop('aircraftregistration', 'starttime')

    # rows with missing values in the label column are filled with 'No Maintenance'
    matrix2 = matrix2.fillna('No Maintenance', subset=['kind'])

    newSchema = StructType([
        StructField("aircraft id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("avg_sensor", DoubleType(), True),
        StructField("flighthours", DoubleType(), True),
        StructField("flightcycles", IntegerType(), True),
        StructField("delayedminutes", IntegerType(), True),
        StructField("kind", StringType(), True)
    ])

    matrix2 = matrix2.withColumn('avg_sensor', matrix2['avg_sensor'].cast(DoubleType())) \
                        .withColumn('flighthours', matrix2['flighthours'].cast(DoubleType())) \
                        .withColumn('flightcycles', matrix2['flightcycles'].cast(IntegerType())) \
                        .withColumn('delayedminutes', matrix2['delayedminutes'].cast(IntegerType()))
    
    matrix2 = spark.createDataFrame(data=matrix2.rdd, schema=newSchema, verifySchema=True)

    return matrix.orderBy('aircraft id', 'date'), matrix2.orderBy('aircraft id', 'date')


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

    labels = labels.groupBy('aircraftregistration', 'starttime').agg(lit('Maintenance').alias('kind'))
    
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

    combined_df = df_list[0]
    for i in range(1, len(df_list)):
        combined_df = combined_df.union(df_list[i])

    result_df = combined_df.groupBy("aircraft id", "date").agg(avg("value").alias("avg_sensor"))

    return result_df



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
    print(sensor_data.count())
    
    print(f'{Fore.YELLOW}Extarcting KPIs data...{Fore.RESET}')
    kpis = extract_dw_data(spark, dbw_properties)
    # print(kpis.count())

    # los labels son mirando a 7 dias vista para cada vuelo. Es decir que un vuelo tiene label maintenance si es que en los 7 dias siguientes tiene un vuelo con label maintenance
    print(f'{Fore.YELLOW}Extarcting maintenance labels...{Fore.RESET}')
    labels = extract_labels(spark, damos_properties)
    # print(labels.count())

    matrix, matrix2 = join_dataframes(spark, sensor_data, kpis, labels)
    print(f'{Fore.GREEN}End of the Managment Pipeline{Fore.RESET}' + '\n' + '-'*50)

    return matrix2