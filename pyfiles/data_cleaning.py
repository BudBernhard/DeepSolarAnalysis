"""
DOCSTRING: This module contains functions to check for multicollinearity, deal with missing values, and other functions to clean the data.

"""

def drop_rates_columns(df):
    """Given that the data has information coded as both rates and absolute values, we separate out the columns and drop the redundant 
    information coded as rates.
    
    df: original dataframe with both absolute values and rates.
    """
    cols = list(df.columns)
    
    # identify rates columns
    rates = []
    for col in cols:
        if 'rate' in col:
            rates.append(col)
    
    # drop the rates columns
    new_df = df.drop(rates, axis = 1)
    return new_df

