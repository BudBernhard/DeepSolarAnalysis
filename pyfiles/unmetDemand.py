"""
DOCSTRING: Fuctions for getting False Positive and reverting to original Scale

"""
import pandas as pd
import numpy as np

def isFalsePositive(originaldf, X_test, y_test, best_clf):
    """originaldf: original dataframe that was split.
        X_test: from original dataframe
        y_test: from original dataframe
        best_clf: This is the gridsearchcv.best_estimator_
    """

    y_pred=best_clf.predict(X_test)
    dfdropped = originaldf.drop('has_tiles', axis = 1)
    test = X_test[(y_test == 0) & (y_pred[:] == 1)]
    falsePositives = pd.DataFrame(test)
    falsePositives = falsePositives.set_axis(dfdropped.columns, axis=1, inplace=False)

    return falsePositives



def isOpportunityZone(falsePositives, scaler = "None"):
    '''Takes false positve results and scaler that was used from a ML model, and compares them to Opportunity Zones, returns a DataFrame of overlapping sub-counties. 
    '''
    
    if (scaler != "None"):
        inversed = scaler.inverse_transform(falsePositives)
        inverseFalsePositives = pd.DataFrame(inversed)
        inverseFalsePositives = inverseFalsePositives.set_axis(falsePositives.columns, axis=1, inplace=False)
    else:
        pass


    ozdf = pd.read_csv("../data/ListOfOppurtunityZonesWithoutAKorHI.csv", encoding = "utf-8")
    ozdf = ozdf.rename(columns={"Census Tract Number": "Census_Tract_Number", "Tract Type": "Tract_Type", "ACS Data Source": "ACS_Data_Source"})
    results = pd.merge(inverseFalsePositives, ozdf, left_on = inverseFalsePositives.fips.astype(np.int64), right_on = ozdf.Census_Tract_Number)
    return results
