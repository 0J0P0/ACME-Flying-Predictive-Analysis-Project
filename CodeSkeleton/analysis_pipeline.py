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
from mlflow.tracking import MlflowClient
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


def select_best_classifier(experiment_id: str, client: MlflowClient = None, experiment_name: str = None) -> PipelineModel:
    """."""

    if client is not None:
        best_model_info = client.search_runs(experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)[0]
        best_model = mlflow.spark.load_model(best_model_info.info.artifact_uri + "/model")
    else:  # Runtime model selection
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)
        best_run = runs[0]
        best_model = mlflow.spark.load_model(best_run.info.artifact_uri + "/model")
    
    return best_model


def log_classifier(experiment_id: str, classifier: PipelineModel, num_features: int, metrics: tuple):
    """
    .
    """

    name = classifier.__class__.__name__

    with mlflow.start_run(experiment_id=experiment_id, run_name=name):
        mlflow.spark.log_model(classifier, 'model', registered_model_name=name)
        mlflow.log_metrics({'accuracy': metrics[0], 'recall': metrics[1]})
        mlflow.log_params({'num_features': num_features})  # faltan hiperparámetros
        mlflow.end_run()


def evaluate_classifier(classifier: PipelineModel, test: DataFrame):
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

    return (acc, recall)


def train_model(data:DataFrame, models: list, k: int = 3, s: int = 69) -> list:
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

    print(f'{Fore.GREEN}Training finished.{Fore.RESET}')

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
        
    return matrix.select('features', 'label'), num_features


def analysis_pipe(matrix: DataFrame, experiment_id: str, client: MlflowClient, s: int = 69) -> tuple:
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
  
    train, test = matrix.randomSplit([0.8, 0.2], seed=s)
    train, train_features = format_matrix(train)
    test, _ = format_matrix(test)

    models = [DecisionTreeClassifier(labelCol='label', featuresCol='features'),
                RandomForestClassifier(labelCol='label', featuresCol='features')]

    classifiers = train_model(train, models)

    print(f'{Fore.YELLOW}Evaluating and logging the classifiers...{Fore.RESET}')
    for c in classifiers:
        metrics = evaluate_classifier(c, test)  # faltan hiperparámetros
        log_classifier(experiment_id, c, train_features, metrics)
    
    print(f'{Fore.GREEN}Evaluation finished.{Fore.RESET}')
    best_classifer = select_best_classifier(experiment_id, client)

    return best_classifer, classifiers