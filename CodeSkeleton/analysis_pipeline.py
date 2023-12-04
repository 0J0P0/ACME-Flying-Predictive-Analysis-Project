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
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

def training(data):
    """
    Trains a set of classifiers to predict unscheduled maintenance for a given aircraft.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs, aswell as the labels and aricraft, day and the average measurements of the 3453 sensor.

    Returns
    -------
    classifiers : list
        List of trained classifiers.
    """
    # Train a decision tree classifier from ml
    dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features")

    print("Optimizing Decision Tree Classifier...")
    # Create a pipeline with the DecisionTreeClassifier
    pipeline = Pipeline(stages=[dt])

    # Define the parameter grid, to adjust the maxDepth and maxBins of the DecisionTreeClassifier
    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10, 15]) \
        .addGrid(dt.maxBins, [120, 140, 160]) \
        .build()

    # Define the evaluator that will be used to evaluate the performance of the model, in this case the accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

    # Create the CrossValidator with a given number of folds
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3)  # You might consider increasing this based on the size of your dataset

    # Run cross-validation and choose the best set of parameters
    cvModel = crossval.fit(data)

    # Get the best model from cross-validation
    bestModel= cvModel.bestModel

    models = [bestModel]

    print("Training complete!")
    # Print the best parameters
    print("Best Max Depth: ", bestModel.stages[0].getMaxDepth())
    print("Best Max Bins: ", bestModel.stages[0].getMaxBins())

    # Train a random forest classifier from ml
    rf = RandomForestClassifier(labelCol="labels", featuresCol="features")

    print("Optimizing Random Forest Classifier...")

    # Create a pipeline with the RandomForestClassifier
    pipeline = Pipeline(stages=[rf])

    # Define the parameter grid, to adjust the maxDepth and maxBins of the RandomForestClassifier

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.maxBins, [120, 140, 160]) \
        .build()
    
    # Define the evaluator that will be used to evaluate the performance of the model, in this case the accuracy

    evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

    # Create the CrossValidator with a smaller number of folds

    # Create the CrossValidator with a given number of folds
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3)  # You might consider increasing this based on the size of your dataset
    
    # Run cross-validation and choose the best set of parameters

    cvModel = crossval.fit(data)

    # Get the best model from cross-validation

    bestModel= cvModel.bestModel

    models.append(bestModel)

    return models

    

def format_data(data):
    """ 
    Formats the data for training.
    """

    # 1. Convert 'day' to a numerical representation
    # data = data.withColumn("day", col("day").cast("string"))
    # data = data.withColumn("aircraft id", col("aircraft id").cast("string"))
    # data = data.withColumn("kind", col("kind").cast("string"))
    data = data.withColumn("avg_sensor", col("avg_sensor").cast("float"))
    data = data.withColumn("flighthours", col("flighthours").cast("float"))
    data = data.withColumn("flightcycles", col("flightcycles").cast("float"))
    data = data.withColumn("delayedminutes", col("delayedminutes").cast("float"))

    # # 2. StringIndexer for converting categorical variables to numerical ones
    indexers = [StringIndexer(inputCol="aircraft id", outputCol="aircraft_id", handleInvalid="skip"),
                StringIndexer(inputCol="day", outputCol="day_id", handleInvalid="skip"),
                StringIndexer(inputCol="kind", outputCol="labels")]

    # indexers = [StringIndexer(inputCol="kind", outputCol="labels")]

    pipeline = Pipeline(stages=indexers)
    data = pipeline.fit(data).transform(data)

    # 3. Assemble feature vector
    assembler = VectorAssembler(inputCols=["aircraft_id", "day_id", "avg_sensor", "flighthours", "flightcycles", "delayedminutes"], outputCol="features")
    data = assembler.transform(data)

    # 4. Select relevant columns (features and label)
    data = data.select("features", "labels")

    return data

def evaluate_classifiers(classifiers: list, test):
    """
    Evaluates the classifiers.

    Parameters
    ----------
    classifiers : list
        List of trained classifiers.
    test : pyspark.sql.DataFrame
        DataFrame with the test data.

    Returns
    -------
    best_classifier : pyspark.ml.PipelineModel
        Best classifier.
    """
    
    best_accuracy = 0

    # Evaluate classifiers
    for classifier in classifiers:
        predictions = classifier.transform(test)
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="labels",
                                                      predictionCol="prediction",
                                                      metricName="accuracy")
        accuracy = evaluator_acc.evaluate(predictions)
        
        print("Accuracy for classifier: ", accuracy)

        evaluator_rec = MulticlassClassificationEvaluator(labelCol="labels",
                                                      predictionCol="prediction",
                                                      metricName="accuracy")
        recall = evaluator_rec.evaluate(predictions)
        
        print("Recall for classifier: ", recall)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = classifier

    # Get the best classifier
    print(Fore.GREEN + "Best classifier: ", best_classifier.stages[0].__class__.__name__, Fore.RESET)

    return best_classifier

def train_classifiers(spark: SparkSession, df):
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

    # #Transform the data to the format needed for training
    df = format_data(df)

    # print('-'*50)
    # print(df.dtypes)
    # print('-'*50)
    # print(df.show())
    # print('-'*50)

    # Do the train and test split of df
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # Train classifiers
    classifiers = training(train)

    # Evaluate classifiers
    best_classifier = evaluate_classifiers(classifiers, test)

    # print(best_classifier)
    # # Save the classiers 
    # best_classifier.write().overwrite().save("models/best_classifier")
    # for classifier in classifiers:
    #     classifier.write().overwrite().save("models/" + classifier.stages[0].__class__.__name__)