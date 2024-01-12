##############################################################################################################
# Authors:      Enric Millan, Juan Pablo Zaldivar                                                            #
#                                                                                                            #
# Project:      Predictive Analysis Project - Classifier Pipeline                                            #
#                                                                                                            #
# Usage:        Called from main.py.                                                                         #
##############################################################################################################

"""
This module accounts for extra functions for general use across the project. 
"""

##############################################################################################################
#                                                                                                            #
# Imports                                                                                                    #
#                                                                                                            #
##############################################################################################################

import argparse
from mlflow.tracking import MlflowClient

##############################################################################################################
#                                                                                                            #
# Functions                                                                                                  #
#                                                                                                            #
##############################################################################################################


def get_model_idx(model: str) -> int:
    """
    Gets the index of the model to use.

    Parameters
    ----------
    model : str
        Model to use.

    Returns
    -------
    model_idx : int
        Index of the model to use.
    """

    if model == 'DecisionTree':
        model_idx = 0
    elif model == 'RandomForest':
        model_idx = 1
    else:
        model_idx = 0
    
    return model_idx


def read_arguments() -> tuple:
    """
    Reads the arguments passed to the script. If no arguments are passed, the default values are used. The required arguments are the database user and password.

    Returns
    -------
    DBuser : str
        Database user.
    DBpassword : str
        Database password.
    python_version : str
        Python version to use.
    stage : str
        Stage to run.
    model : int
        Model to use.

    Raises
    ------
    argparse.ArgumentError
        If the arguments are not valid.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--DBuser',
                        required=True,
                        default=None,
                        help='Database user')
    parser.add_argument('--DBpassword',
                        required=True,
                        default=None,
                        help='Database password')
    
    parser.add_argument('--python_version',
                        required=False,
                        default='11',
                        choices=['10', '11'],
                        help='Database password')
    
    parser.add_argument('--stage',
                        required=False,
                        default='all',
                        choices=['all', 'management', 'analysis', 'classifier'],
                        help='Stage to run')
    
    parser.add_argument('--model',
                        required=False,
                        default='Default',
                        choices=['DecisionTree', 'RandomForest', 'Default'],
                        help='Model for prediction (best model selected by default)')

    args = parser.parse_args()

    return args.DBuser, args.DBpassword, args.python_version, args.stage, get_model_idx(args.model)


def create_or_load_experiment(client: MlflowClient, experiment_name: str) -> str:
    """
    Creates or loads an experiment in MLflow. If the experiment already exists, it is loaded.

    Parameters
    ----------
    client : MlflowClient
        MLflow client.
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    experiment_id : str
        Experiment ID.
    """

    experiment_description = (
        "Predictive Analysis Project - Analysis Pipeline"
    )

    experiment_tags = {
        "project_name": "Predictive Analysis Project",
        "team": "Juan Pablo Zaldivar, Enric Millan",
        "mlflow.note.content": experiment_description,
    }

    try:
        experiment_id = client.create_experiment(name=experiment_name, tags=experiment_tags)
    except :
        experiment_id = client.get_experiment_by_name(name=experiment_name).experiment_id
    
    return experiment_id