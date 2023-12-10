##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Analysis Pipeline                                              #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################


"""
This pipeline trains a set of classifiers to predict unscheduled maintenance for a given aircraft. The pipeline is composed of the following steps:

    - Format the data.
    - Train the classifiers.
    - Evaluate the classifiers.
    - Log the metrics and save the classifiers.
"""

##############################################################################################################
#                                                                                                            #
# Imports                                                                                                    #
#                                                                                                            #
##############################################################################################################

import mlflow
from colorama import Fore
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################

def evaluate_classifiers(classifiers: list, test: DataFrame):
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


def evaluate_and_log_metrics(classifier: PipelineModel, test: DataFrame):
    """
    ...
    """

    evaluator1 = MulticlassClassificationEvaluator(labelCol='label',
                                                   predictionCol='prediction',
                                                   metricName='accuracy')
    
    acc = evaluator1.evaluate(classifier.transform(test))
    mlflow.log_metrics({'accuracy': acc})

    evaluator2 = MulticlassClassificationEvaluator(labelCol='label',
                                                   predictionCol='prediction',
                                                   metricName='weightedRecall')
    
    recall = evaluator2.evaluate(classifier.transform(test))
    mlflow.log_metrics({'recall': recall})


def train_model(data:DataFrame, models: list, k: int = 3, s: int = 42) -> list:
    """
    ---
    """

    classifiers = []

    for m in models:
        print(f'{Fore.YELLOW}Training {m.__class__.__name__}...{Fore.RESET}')

        # pipeline = Pipeline(stages=[m])  # es util el pipeline si solo hay un stage?

        paramGrid = ParamGridBuilder() \
            .addGrid(m.maxDepth, [5, 10, 15]) \
            .addGrid(m.maxBins, [120, 140, 160]) \
            .build()
        
        evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                                      predictionCol='prediction',
                                                      metricName='accuracy')
        
        cv = CrossValidator(estimator=m, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=k, seed=s)
        
        cvModel = cv.fit(data)
        
        classifiers.append(cvModel.bestModel)

    return classifiers


def format_matrix(matrix: DataFrame) -> DataFrame:
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

    aircraftIndexer = StringIndexer(inputCol='aircraft id', outputCol='aircraft_id')
    dateIndexer = StringIndexer(inputCol='date', outputCol='date_id')
    assembler = VectorAssembler(inputCols=['aircraft_id', 'date_id', 'avg_sensor', 'flighthours', 
                                           'flightcycles', 'delayedminutes'],
                                outputCol='features')

    pipeline = Pipeline(stages=[aircraftIndexer, dateIndexer, assembler])
    matrix = pipeline.fit(matrix).transform(matrix)

    num_features = len(matrix.select('features').first()[0])
    
    # print(matrix.select('features', 'label').show(5))
    return matrix.select('features', 'label'), num_features


def analysis_pipe(matrix: DataFrame, experiment_name: str = 'TrainClassifiers', s: int = 42):
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
  
    matrix, num_features = format_matrix(matrix)
    
    mlflow.set_experiment(experiment_name)

    train, test = matrix.randomSplit([0.8, 0.2], seed=s)

    with mlflow.start_run():

        models = [DecisionTreeClassifier(labelCol='label', featuresCol='features'),
                  RandomForestClassifier(labelCol='label', featuresCol='features')]

        classifiers = train_model(train, models, 3)

        for c in classifiers:
            # model_name = c.stages[0].__class__.__name__
            model_name = c.__class__.__name__
            
            mlflow.spark.log_model(c, model_name)
            mlflow.log_params({'num_features': num_features})
            
            evaluate_and_log_metrics(c, test)
            mlflow.spark.save_model(c, model_name)

    mlflow.end_run()

    # best_classifier = evaluate_classifiers(classifiers, test)
    # return best_classifier, classifiers

    return classifiers[0], classifiers