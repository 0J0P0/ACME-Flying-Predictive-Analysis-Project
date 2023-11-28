import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession
from management_pipeline import managment_pipe
from analysis_pipeline import train_classifiers


HADOOP_HOME = './resources/hadoop_home'
JDBC_JAR = './resources/postgresql-42.2.8.jar'
PYSPARK_PYTHON = 'python3.11'
PYSPARK_DRIVER_PYTHON = 'python3.11'


if __name__== '__main__':
    os.environ['HADOOP_HOME'] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + '\\bin')
    os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
    os.environ['PYSPARK_DRIVER_PYTHON'] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set('spark.master', 'local')
    conf.set('spark.app.name','DBALab')
    conf.set('spark.jars', JDBC_JAR)

    # Initialize a Spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Create and point to your pipelines here
    dbw_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require',
                 'user': 'juan.pablo.zaldivar',
                 'password': 'DB021202'}

    damos_properties = {'driver': 'org.postgresql.Driver',
                 'url': 'jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require',
                 'user': 'juan.pablo.zaldivar',
                 'password': 'DB021202'}
    
    # matrix = managment_pipe('./resources/trainingData/', spark, dbw_properties, damos_properties)
    if not os.path.exists('./resources/matrix'):
        matrix, matrix2 = managment_pipe('./resources/trainingData/', spark, dbw_properties, damos_properties)
        matrix2.write.csv('./resources/matrix', header=True)
    else:
        matrix2 = spark.read.csv('./resources/matrix', header=True)

    # train_classifiers(spark, matrix2)