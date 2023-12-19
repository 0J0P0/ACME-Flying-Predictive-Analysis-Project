##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Main                                                           #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################


"""
This is the main file of the project. It is used to call the different pipelines.

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
from analysis_pipeline import analysis_pipe
from classifier_pipeline import classifier_pipe, read_saved_model
from management_pipeline import managment_pipe, read_saved_matrix

##############################################################################################################
#                                                                                                            #
# Variables                                                                                                  #
#                                                                                                            #
##############################################################################################################

HADOOP_HOME = './resources/hadoop_home'
JDBC_JAR = './resources/postgresql-42.2.8.jar'
PYSPARK_PYTHON = 'python3.11'
PYSPARK_DRIVER_PYTHON = 'python3.11'

dbw_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require',
                 'user': 'juan.pablo.zaldivar',
                 'password': 'DB021202'}

amos_properties = {'driver': 'org.postgresql.Driver',
                'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require',
                'user': 'juan.pablo.zaldivar',
                'password': 'DB021202'}

# dbw_properties = {'driver': 'org.postgresql.Driver',
#              'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require',
#              'user': 'enric.millan.iglesias',
#              'password': 'DB220303'}

# amos_properties = {'driver': 'org.postgresql.Driver',
#              'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require',
#              'user': 'enric.millan.iglesias',
#              'password': 'DB220303'}


def read_saved_pipelines(spark: SparkSession):
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
            return read_saved_matrix(spark), read_saved_model()
        except:
            print(f'{Fore.RED}Error reading the matrix and/or the model from the resources folder. Try executing the management and analysis pipelines first.{Fore.RESET}')
            sys.exit(1)


def pipeline_stage():
    """
    .
    """

    try:
        stage = sys.argv[1]
        if stage not in ['all', 'management', 'analysis', 'classifier']:
            print(f'{Fore.RED}Invalid stage argument, try any of the following: all, management, analysis, classifier{Fore.RESET}')
            sys.exit(1)
    except IndexError:
        stage = 'all'
        print(f'{Fore.YELLOW}No stage argument provided, running all stages{Fore.RESET}')

    return stage


##############################################################################################################
#                                                                                                            #
# Main                                                                                                       #
#                                                                                                            #
##############################################################################################################

if __name__== '__main__':
    os.environ['HADOOP_HOME'] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + '\\bin')
    os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
    os.environ['PYSPARK_DRIVER_PYTHON'] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()
    conf.set('spark.master', 'local[*]')
    conf.set('spark.app.name','DBALab')
    conf.set('spark.jars', JDBC_JAR)

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    stage = pipeline_stage()

    if stage == 'all':
        print('-'*50 + '\n' + f'{Fore.CYAN}Start of the Management Pipeline{Fore.RESET}')
        matrix = managment_pipe(spark, dbw_properties, amos_properties, stage)
        print(f'{Fore.GREEN}End of the Management Pipeline{Fore.RESET}' + '\n' + '-'*50)

    if stage == 'all' or stage == 'analysis':
        print(f'{Fore.CYAN}Start of the Analysis Pipeline{Fore.RESET}')
        if stage == 'analysis':
            matrix = read_saved_pipelines(spark)
        model, _ = analysis_pipe(matrix)
        print(f'{Fore.GREEN}End of the Analysis Pipeline{Fore.RESET}' + '\n' + '-'*50)
    
    print(f'{Fore.CYAN}Start of the Classifier Pipeline{Fore.RESET}')
    if stage == 'classifier':
        # choose model 
        matrix, model = read_saved_pipelines(spark)
        # print(matrix)
    classifier_pipe(model, matrix)
    print(f'{Fore.GREEN}End of the Classifier Pipeline{Fore.RESET}' + '\n' + '-'*50)


