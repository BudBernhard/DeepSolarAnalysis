"""
DOCSTRING: This module contains the different values we used in gridsearch to optimize hyperparamters for various models. 

"""

from sklearn.pipeline import Pipeline



# Decision Tree Classifier
def decision_tree_parameters(X_train, y_train):
    """This function performs a grid search on the hyperparamters for a decision tree classifier, and returns the optimal parameters."""
    parameters = {'max_depth':[2, 3, 4, 5, 8], 'min_samples_split':[2, 5, 10, 15, 100], 'min_samples_leaf':[1, 2, 5, 10]}
    dt = GridSearchCV(DecisionTreeClassifier(), parameters, cv = 5)
    dt.fit(X_train, ytrain)
    return dt.best_params_