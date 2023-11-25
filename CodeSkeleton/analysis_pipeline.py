"""
This pipeline trains a set of classifiers to predict unscheduled maintenance for a given aircraft. The pipeline is composed of the following steps:
- Train and test split
- Format data for training
- Train classifiers
- Evaluate classifiers
- Save classifiers
"""


import os
from colorama import Fore
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import avg, sum, lit, date_format, to_date, substring

def train_classifiers(df: DataFrame):
    """
    Trains a set of classifiers to predict unscheduled maintenance for a given aircraft.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs, aswell as the labels and the average measurements of the 3453 sensor.

    Returns
    -------
    classifiers : list
        List of trained classifiers.
    """
    # Train a decision tree classifier from ml
    from pyspark.ml.classification import DecisionTreeClassifier
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # Train a random forest classifier from ml
    from pyspark.ml.classification import RandomForestClassifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    return [dt, rf]
    return classifiers
def train_classifiers(spark: SparkSession, df: pyspark.sql.DataFrame):
    """
    Trains a set of classifiers to predict unscheduled maintenance for a given aircraft.

    Parameters
    ----------
    spark : SparkSession
        SparkSession object.
    df : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs, aswell as the labels and the average measurements of the 3453 sensor.

    Returns
     The trained classifiers, making a differenciation of the one with the best performance.
    -------
    None
    """

    # Do the train and test split of df
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Train classifiers
    classifiers = train_classifiers(df)

    # Evaluate classifiers
    evaluate_classifiers(classifiers, df)

    # Save classifiers
    save_classifiers(classifiers, aircraft_registration)

    return None