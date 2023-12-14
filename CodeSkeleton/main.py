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
from management_pipeline import managment_pipe
from analysis_pipeline import analysis_pipe
from classifier_pipeline import classifier_pipe

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


    print('-'*50 + '\n' + f'{Fore.CYAN}Start of the Management Pipeline{Fore.RESET}')
    
    matrix = managment_pipe(spark, dbw_properties, amos_properties)
    
    print(f'{Fore.GREEN}End of the Management Pipeline{Fore.RESET}' + '\n' + '-'*50 + '\n' + f'{Fore.CYAN}Start of the Analysis Pipeline{Fore.RESET}')
    
    model, _ = analysis_pipe(matrix)
    
    print(f'{Fore.GREEN}End of the Analysis Pipeline{Fore.RESET}' + '\n' + '-'*50 + '\n' + f'{Fore.CYAN}Start of the Classifier Pipeline{Fore.RESET}')

    classifier_pipe(model, matrix)

    print(f'{Fore.GREEN}End of the Classifier Pipeline{Fore.RESET}' + '\n' + '-'*50)
