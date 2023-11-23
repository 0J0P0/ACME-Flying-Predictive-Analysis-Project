"""
This pipeline trains a set of classifiers to predict unscheduled maintenance for a given aircraft. The pipeline is composed of the following steps:
- Data ingestion
- Train and test split
- Format data for training
- Train classifiers
- Evaluate classifiers
- Save classifiers
"""


import os
from colorama import Fore
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, sum, lit, date_format, to_date, substring