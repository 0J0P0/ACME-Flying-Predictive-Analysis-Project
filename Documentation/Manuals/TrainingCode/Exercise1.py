from pyspark import SparkConf
from pyspark.sql import SparkSession


sc = SparkConf()\
    .set("spark.master", "local")\
        .set("spark.app.name", "DBALab")


def process(sc):
    spark = SparkSession.builder.config(conf=sc).getOrCreate()

    df = spark.read.csv('resources/bank.csv', header=True, sep=";", quote="\"", escape="\"")
    # df.show()

    df.createOrReplaceTempView("bank")
    spark.sql('SELECT job, loan, AVG(balance) FROM bank GROUP BY job, loan ORDER BY job, loan').show()

    

    spark.stop()

process(sc)
