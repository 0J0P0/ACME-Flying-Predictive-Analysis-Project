"""
This pipeline trains a set of classifiers to predict unscheduled maintenance for a given aircraft. The pipeline is composed of the following steps:
- Train and test split
- Format data for training
- Train classifiers
- Evaluate classifiers
- Save classifiers
"""

import mlflow
from colorama import Fore
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier


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
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="label",
                                                      predictionCol="prediction",
                                                      metricName="accuracy")
        accuracy = evaluator_acc.evaluate(predictions)
        
        print("Accuracy for classifier: ", accuracy)

        #weighted to take into account the imbalance of classes
        evaluator_rec = MulticlassClassificationEvaluator(labelCol="label",
                                                      predictionCol="prediction",
                                                      metricName="weightedRecall")
        recall = evaluator_rec.evaluate(predictions)
        
        print("Recall for classifier: ", recall)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = classifier

    # Get the best classifier
    print(Fore.GREEN + "Best classifier: ", best_classifier.stages[0].__class__.__name__, Fore.RESET)

    return best_classifier


def training(data):
    """
    Trains a set of classifiers to predict unscheduled maintenance for a given aircraft.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs, aswell as the label and aricraft, day and the average measurements of the 3453 sensor.

    Returns
    -------
    classifiers : list
        List of trained classifiers.
    """

    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    print("Optimizing Decision Tree Classifier...")
    # Create a pipeline with the DecisionTreeClassifier
    pipeline = Pipeline(stages=[dt])

    # Define the parameter grid, to adjust the maxDepth and maxBins of the DecisionTreeClassifier
    paramGrid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10, 15]) \
        .addGrid(dt.maxBins, [120, 140, 160]) \
        .build()

    # Define the evaluator that will be used to evaluate the performance of the model, in this case the accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

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
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    print("Optimizing Random Forest Classifier...")

    # Create a pipeline with the RandomForestClassifier
    pipeline = Pipeline(stages=[rf])

    # Define the parameter grid, to adjust the maxDepth and maxBins of the RandomForestClassifier

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.maxBins, [120, 140, 160]) \
        .build()
    
    # Define the evaluator that will be used to evaluate the performance of the model, in this case the accuracy

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

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


def format_data(matrix: DataFrame) -> DataFrame:
    """ 
    Formats the matrix for training. Converts the categorical variables to numerical ones and creates a vector with the features.

    Parameters
    ----------
    matrix : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs, aswell as the label and aricraft, day and the average measurements of the 3453 sensor.
    
    Returns
    -------
    matrix : pyspark.sql.DataFrame
        DataFrame with the formatted matrix.
    """

    indexers = [StringIndexer(inputCol='aircraft id', outputCol='aircraft_id', handleInvalid='skip'),
                StringIndexer(inputCol='date', outputCol='date_id', handleInvalid='skip')]

    pipeline = Pipeline(stages=indexers)
    matrix = pipeline.fit(matrix).transform(matrix)

    assembler = VectorAssembler(inputCols=['aircraft_id', 'date_id', 'avg_sensor', 'flighthours', 'flightcycles', 'delayedminutes'],
                                outputCol='features')
    
    matrix = assembler.transform(matrix)

    num_features = len(matrix.select('features').first()[0])
    
    # print(matrix.select('features', 'label').show(5))
    return matrix.select('features', 'label'), num_features


def evaluate_and_log_metrics(classifier, test):
    evaluator1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator1.evaluate(classifier.transform(test))
    mlflow.log_metrics({"accuracy": accuracy})

    evaluator2 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    recall = evaluator2.evaluate(classifier.transform(test))
    mlflow.log_metrics({"recall": recall})


def train_classifiers(matrix: DataFrame, experiment_name: str = "TrainClassifiers"):
    """
    Trains a set of classifiers to predict unscheduled maintenance for a given aircraft.

    Parameters
    ----------
    matrix : pyspark.sql.DataFrame
        DataFrame with the FH, FC and DM KPIs, as well as the label and the average measurements of the 3453 sensor.
    experiment_name : str, optional
        Name of the MLflow experiment, by default "TrainClassifiers".

    Returns
    -------
    None
    """
  
    matrix, num_features = format_data(matrix)
    mlflow.set_experiment(experiment_name)

    train, test = matrix.randomSplit([0.8, 0.2], seed=42)

    with mlflow.start_run():
        classifiers = training(train)
        for classifier in classifiers:
            model_name = "model_" + classifier.stages[0].__class__.__name__
            mlflow.spark.log_model(classifier, model_name)
            mlflow.log_params({"num_features": num_features})
            evaluate_and_log_metrics(classifier, test)
            mlflow.spark.save_model(classifier, model_name)

    mlflow.end_run()
    best_classifier = evaluate_classifiers(classifiers, test)

    # print(best_classifier)
    # # Save the classiers 
    # best_classifier.write().overwrite().save("models/best_classifier")
    # for classifier in classifiers:
    #     classifier.write().overwrite().save("models/" + classifier.stages[0].__class__.__name__)