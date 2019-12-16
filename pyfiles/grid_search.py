"""
DOCSTRING: This module contains the different values we used in gridsearch to optimize hyperparamters for various models. 

"""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn..svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



pipe = Pipeline([('classifier', DecisionTreeClassifier())])

search_space = [{'classifier': [DecisionTreeClassifier()],
                 'classifier__max_depth': [2, 3, 4, 5, 8],
                 'classifier__min_samples_split': [2, 5, 10, 15, 100]
                'classifier__min_samples_leaf':[1, 2, 5, 10]},
                {'classifier': [RandomForestClassifier()],
                 'classifier__n_estimators': [10, 100, 1000],
                 'classifier__max_depth': [5, 8, 15, 25, 30], 
                 'classifier__min_samples_split':[1, 2, 5, 10, 15, 100],
                 'classifier__min_samples_leaf': [1, 2, 5, 10],
                 'classifier__max_features': ['log2', 'sqrt', None]}
               {'classifier': [SVC()],
               'classifier__C': [.001, .01, 0.1, 1, 10, 100, 1000],
                   'classifier__gamma': ['auto','scale'],
                   'classifier__class_weight':['balanced', None]}
               {'classifier':[KNeighborsClassifier()],
                   'classifier__n_neighbors':[2,4,8,16],
                   'classifier__p':[2,3]}]

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)

best_model = clf.fit(X_train, y_train)

best_model.best_estimator_.get_params()['classifier']