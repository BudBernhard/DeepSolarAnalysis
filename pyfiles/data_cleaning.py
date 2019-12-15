"""
DOCSTRING: This module contains functions to check for multicollinearity, deal with missing values, and other functions to clean the data.

"""

def drop_rates_and_redundant_columns(df):
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
    
    # Remove all unique identifiers, objects, and booleans
    new_df = new_df.drop(columns=['county',
                          'state',
                         'electricity_price_transportation',
                         'voting_2016_dem_win',
                         'voting_2012_dem_win'])
    return new_df

def create_has_tiles_target_column(df):
    """This function creates our target column, a binary column indicating whether or not there are solar panels in the tract."""
    
    df['has_tiles'] = (df.total_panel_area > 0).mul(1)
    
    # Remove all deepsolar inputs

    df = df.drop(columns=['solar_system_count'], axis = 1)
    df = df.drop(columns=['total_panel_area'], axis = 1)
    df = df.drop(columns=['solar_panel_area_per_capita'], axis =1)

    df = df.drop(columns=['solar_panel_area_divided_by_area'], axis = 1)
    df = df.drop(columns=['tile_count_residential'], axis = 1)
    df = df.drop(columns=['tile_count_nonresidential'], axis = 1)
    df = df.drop(columns=['solar_system_count_residential'], axis =1)

    df = df.drop(columns=['solar_system_count_nonresidential'], axis = 1)
    df = df.drop(columns=['total_panel_area_residential'], axis = 1)
    df = df.drop(columns=['total_panel_area_nonresidential'], axis = 1)
    df = df.drop(columns=['number_of_solar_system_per_household'], axis =1)
    
    return df


    