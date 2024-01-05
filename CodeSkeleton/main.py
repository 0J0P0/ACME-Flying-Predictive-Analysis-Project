##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Main                                                           #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################


"""
This is the main file of the project. It is used to call the different pipelines. 
Its operation requires a sequence of parameters to define necessary configuration settings and credentials to access the databases.
    - Database User: The username or identifier used to access the database.
    - Database Password: The password associated with the provided database user.
    - Python version: The specific version of Python intended to be used.
    - Pipeline stage: Specifies the stage in the pipeline to be executed (options: 'all', 'management', 'analysis', 'classifier').
    - Model: Indicates the type of model to be utilized for prediction (options: 'DecisionTree', 'RandomForest', 'Default'). The 'Default' option represents the best-selected model by default.

Usage
-----
python main.py
"""

##############################################################################################################
#                                                                                                            #
# Imports                                                                                                    #
#                                                                                                            #
##############################################################################################################

import os
import sys
from colorama import Fore
from pyspark import SparkConf
from pyspark.sql import SparkSession
from utils import read_arguments
from analysis_pipeline import analysis_pipe
from classifier_pipeline import classifier_pipe, read_saved_model
from management_pipeline import managment_pipe, read_saved_matrix

##############################################################################################################
#                                                                                                            #
# Variables                                                                                                  #
#                                                                                                            #
##############################################################################################################

def read_saved_pipelines(spark: SparkSession, model_name: str = None):
    """
    .
    """

    if stage == 'analysis':
        try:
            return read_saved_matrix(spark)
        except:
            print(f'{Fore.RED}Error reading the matrix from the resources folder. Try executing the management pipeline first.{Fore.RESET}')
            sys.exit(1)

    if stage == 'classifier':
        try:
            return read_saved_matrix(spark), read_saved_model(model_name)
        except Exception as e:
            print(f'{Fore.RED}Error reading the matrix and/or the model from the resources folder. Try executing the management and analysis pipelines first: {e}{Fore.RESET}')
            sys.exit(1)


##############################################################################################################
#                                                                                                            #
# Main                                                                                                       #
#                                                                                                            #
##############################################################################################################

if __name__== '__main__':
    user, password, python_version, stage, model_name = read_arguments()
    
    HADOOP_HOME = './resources/hadoop_home'
    JDBC_JAR = './resources/postgresql-42.2.8.jar'
    PYSPARK_PYTHON = f'python3.{python_version}'
    PYSPARK_DRIVER_PYTHON = f'python3.{python_version}'

    os.environ['HADOOP_HOME'] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + '\\bin')
    os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
    os.environ['PYSPARK_DRIVER_PYTHON'] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()
    conf.set('spark.master', 'local[*]')
    conf.set('spark.app.name','DBALab')
    conf.set('spark.jars', JDBC_JAR)
   

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel('OFF')
    dbw_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require',
                 'user': f'{user}',
                 'password': f'{password}'}

    amos_properties = {'driver': 'org.postgresql.Driver',
                'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require',
                'user': f'{user}',
                'password': f'{password}'}


    if stage == 'all':
        print('-'*50 + '\n' + f'{Fore.CYAN}Start of the Management Pipeline{Fore.RESET}')
        matrix = managment_pipe(spark, dbw_properties, amos_properties)
        print(f'{Fore.GREEN}End of the Management Pipeline{Fore.RESET}' + '\n' + '-'*50)

    if stage == 'all' or stage == 'analysis':
        print(f'{Fore.CYAN}Start of the Analysis Pipeline{Fore.RESET}')
        if stage == 'analysis':
            matrix = read_saved_pipelines(spark)
        model, _ = analysis_pipe(matrix)
        print(f'{Fore.GREEN}End of the Analysis Pipeline{Fore.RESET}' + '\n' + '-'*50)
    
    print(f'{Fore.CYAN}Start of the Classifier Pipeline{Fore.RESET}')
    if stage == 'classifier':
        matrix, model = read_saved_pipelines(spark, model_name)
    classifier_pipe(spark, model, dbw_properties)
    print(f'{Fore.GREEN}End of the Classifier Pipeline{Fore.RESET}' + '\n' + '-'*50)


