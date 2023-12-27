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


def evaluate_and_log_metrics(classifier: PipelineModel, test: DataFrame):
    """
    ...
    """

    evaluator1 = MulticlassClassificationEvaluator(labelCol='label',
                                                   predictionCol='prediction',
                                                   metricName='accuracy')
    
    acc = evaluator1.evaluate(classifier.transform(test))

    evaluator2 = MulticlassClassificationEvaluator(labelCol='label',
                                                   predictionCol='prediction',
                                                   metricName='weightedRecall')
    
    recall = evaluator2.evaluate(classifier.transform(test))

    return acc, recall


def train_model(data:DataFrame, models: list, k: int = 3, s: int = 42) -> list:
    """
    ---
    """

    classifiers = []

    for m in models:
        print(f'{Fore.YELLOW}Training {m.__class__.__name__}...{Fore.RESET}')

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
    best_classifier : pyspark.ml.PipelineModel
        Best classifier.
    classifiers : list
        List of trained classifiers.
    """
  
    mlflow.set_experiment(experiment_name)

    train, test = matrix.randomSplit([0.8, 0.2], seed=s)
    train, train_features = format_matrix(train)
    test, _ = format_matrix(test)

    with mlflow.start_run():

        models = [DecisionTreeClassifier(labelCol='label', featuresCol='features'),
                  RandomForestClassifier(labelCol='label', featuresCol='features')]

        classifiers = train_model(train, models)
        sorted_classifiers = []
        
        for c in classifiers:
            model_name = c.__class__.__name__
            
            c_info = mlflow.spark.log_model(c, model_name)

            acc, rec = evaluate_and_log_metrics(c, test)
            mlflow.log_metrics({'num_features': train_features, 'accuracy': acc, 'recall': rec})

            sorted_classifiers.append((c, c_info, acc, rec))

            mlflow.spark.save_model(c, 'models/' + model_name)

    mlflow.end_run()

    sorted_classifiers.sort(key=lambda x: x[2], reverse=True)

    with open('models/classifiers.txt', 'w') as f:
        for c in sorted_classifiers:
            f.write(f'{c[0].__class__.__name__},{c[1].model_uri},{c[2]},{c[3]}\n')

    return sorted_classifiers[0][0], sorted_classifiers