"""
DOCSTRING:

This module contains functions and pipelines for different classification models and parameter gridsearches.

"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# def print_metrics(y_test, y_pred):
#     """This function prints out a list of precision, recall, accuracy, and f1-scores for the model."""
#     return ("Precision: {}".format(precision_score(y_test, y_pred))), "Recall: {}".format(recall_score(y_test, y_pred))), "Accuracy: {}".format(accuracy_score(y_test, y_pred))), "F1 Score: {}".format(f1_score(y_test, y_pred))))


def pick_sampling_method(X_train, y_train, method = ['oversampling', 'undersampling', 'smote']):
    
    ros = RandomOverSampler()
    rus = RandomUnderSampler()
    smt = SMOTE()
    
    if method == 'oversampling':
        X, y = ros.fit_resample(X_train, y_train)
    elif method == 'undersampling':
        X, y = rus.fit_resample(X_train, y_train)
    else:
        X, y = smt.fit_sample(X_train, y_train)
    
    return X, y