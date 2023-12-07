import os
import sys
from colorama import Fore
from pyspark import SparkConf
from pyspark.sql import SparkSession
from management_pipeline import managment_pipe
from analysis_pipeline import train_classifiers


HADOOP_HOME = './resources/hadoop_home'
JDBC_JAR = './resources/postgresql-42.2.8.jar'
PYSPARK_PYTHON = 'python3.10'
PYSPARK_DRIVER_PYTHON = 'python3.10'


if __name__== '__main__':
    os.environ['HADOOP_HOME'] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + '\\bin')
    os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
    os.environ['PYSPARK_DRIVER_PYTHON'] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set('spark.master', 'local')
    conf.set('spark.app.name','DBALab')
    conf.set('spark.jars', JDBC_JAR)

    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    """ dbw_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require',
                 'user': 'juan.pablo.zaldivar',
                 'password': 'DB021202'}

    damos_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require',
                 'user': 'juan.pablo.zaldivar',
                 'password': 'DB021202'} """
    
    dbw_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require',
                 'user': 'enric.millan.iglesias',
                 'password': 'DB220303'}

    damos_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require',
                 'user': 'enric.millan.iglesias',
                 'password': 'DB220303'}
    

    print('-'*50 + '\n' + f'{Fore.CYAN}Start of the Management Pipeline{Fore.RESET}')
    
    matrix = managment_pipe('./resources/trainingData/', spark, dbw_properties, damos_properties)
    
    print(f'{Fore.GREEN}End of the Management Pipeline{Fore.RESET}' + '\n' + '-'*50 + '\n' + f'{Fore.CYAN}Start of the Analysis Pipeline{Fore.RESET}')
    
    train_classifiers(matrix)
    
    print(f'{Fore.GREEN}End of the Analysis Pipeline{Fore.RESET}' + '\n' + '-'*50)
