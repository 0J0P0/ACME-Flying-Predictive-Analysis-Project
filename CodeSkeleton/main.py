import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from management_pipeline import extract_sensor_data


HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.11"
PYSPARK_DRIVER_PYTHON = "python3.11"


if __name__== "__main__":
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.master", "local")
    conf.set("spark.app.name","DBALab")
    conf.set("spark.jars", JDBC_JAR)

    # Initialize a Spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Create and point to your pipelines here
    sensor_data = extract_sensor_data('resources/trainingData/', spark)

    print(type(sensor_data))

    # sensor_data.show()



