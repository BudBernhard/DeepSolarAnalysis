"""
DOCSTRING: This module contains the different values we used in gridsearch to optimize hyperparamters for various models. 

"""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def find_hyperparameters(pipe, params, X_train, y_train):
    
    res = GridSearchCV(pipe, params, scoring = 'f1', return_train_score = True)
    res.fit(X_train, y_train)
    
    return res




pipe_dt = Pipeline([('dt', DecisionTreeClassifier())])
params_dt = {'dt__max_depth': [2, 4, 8],
                 'dt__min_samples_split': [2, 5, 10],
                'dt__min_samples_leaf':[1, 5, 10]}


pipe_rf = Pipeline([('rf', RandomForestClassifier())])
params_rf = {'classifier__n_estimators': [10, 100, 1000],
                 'classifier__max_depth': [5, 8, 15, 25, 30], 
                 'classifier__min_samples_split':[2, 5, 10, 15, 100],
                 'classifier__min_samples_leaf': [1, 2, 5, 10],
                 'classifier__max_features': ['log2', 'sqrt', None]}

pipe_svc = Pipeline([('svc', SVC())])
params_svc = {'classifier__C': [.001, .01, 0.1, 1, 10, 100, 1000],
                   'classifier__gamma': ['auto','scale'],
                   'classifier__class_weight':['balanced', None]},

pipe_knn = Pipeline([('classifier', KNeighborsClassifier())]),
params_knn = {'classifier__n_neighbors':[2,4,8,16],
                'classifier__p':[2,3]}
