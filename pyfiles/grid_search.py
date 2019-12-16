"""
DOCSTRING: This module contains the different values we used in gridsearch to optimize hyperparamters for various models. 

"""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn..svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Decision Tree Classifier
def tune_decision_tree_hyperparameters(X_train, y_train):
    """This function performs a grid search on the hyperparamters for a decision tree classifier, and returns the optimal parameters."""
    
    parameters = {'max_depth':[2, 3, 4, 5, 8], 'min_samples_split':[2, 5, 10, 15, 100], 'min_samples_leaf':[1, 2, 5, 10]}
    dt = GridSearchCV(DecisionTreeClassifier(), parameters, cv = 5)
    dt.fit(X_train, y_train)
    return dt.best_params_


def tune_random_forest_hyperparameters(X_train, y_train):
    """This function performs a grid search on the hyperparamters for a random forests classifier and returns the optimal parameters."""
    
    params ={'n_estimators': [120, 300, 500, 800], 'max_depth':[5, 8, 15, 25, 30], 'min_samples_split':[1, 2, 5, 10, 15, 100], 'min_samples_leaf'=[1, 2, 5, 10], 'max_features'= ['log2', 'sqrt', None]
    
    rf = GridSearchCV(RandomForestClassifier(), params, cv=5)
    rf.fit(X_train, y_train)
    
    return rf.best_params_
             
             
def tune_svm_hyperparameters(X_train, y_train):
    """This function performs a grid search on the hyperparameters for an SVM model and returns the optimal parameters."""
    params = {'C':[.001, .01, 0.1, 1, 10, 100, 1000], 'gamma': ['auto','scale'], 'class_weight' : ['balanced', None]}
             
    svm = GridSearchCV(SVC(), params, cv=5)
    svm.fit(X_train, y_train)
    return svm.best_params_
             
             
def tune_knn_hyperparameters(X_train, y_train):
     """This function performs a grid search on the hyperparameters for a KNN model and returns the optimal parameters."""
    params = {'n_neighbors': [2, 4, 8, 16], 'p': [2,3]}
    knn = GridSearchCV(KNeighborsClassifier(), params, cv = 5)
    return knn.best_params_