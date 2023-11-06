import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import Exercise1
import Exercise2
import Exercise3

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.6"
PYSPARK_DRIVER_PYTHON = "python3.6"

if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    if(len(sys.argv) < 2):
        print("Wrong number of parameters, usage: (exercise1, exercise2, exercise3)")
        exit()
    if(sys.argv[1] == "exercise1"):
        Exercise1.process(spark)
    elif(sys.argv[1] == "exercise2"):
        Exercise2.process(spark)
    elif(sys.argv[1] == "exercise3"):
        Exercise3.process(spark)
    else:
       print("Wrong exercise number")