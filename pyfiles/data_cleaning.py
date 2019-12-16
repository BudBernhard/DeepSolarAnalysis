"""
DOCSTRING: This module contains functions to check for multicollinearity, deal with missing values, and other functions to clean the data.

"""

def drop_redundant_columns(df):
    """Given that the data has information coded as both rates and absolute values, we identified and dropped redundant 
    information.
    
    df: original dataframe with both absolute values and rates.
    """
    cols = list(df.columns)
    
    # identify redundant columns
    redundant = ['education_less_than_high_school_rate',
     'education_high_school_graduate_rate',
     'education_college_rate',
     'education_bachelor_rate',
     'education_master_rate',
     'education_professional_school_rate',
     'education_doctoral_rate',
     'race_white_rate',
     'race_black_africa_rate',
     'race_indian_alaska_rate',
     'race_asian_rate',
     'race_islander_rate',
     'race_other_rate',
     'race_two_more_rate','heating_fuel_gas_rate',
     'heating_fuel_electricity_rate',
     'heating_fuel_fuel_oil_kerosene_rate',
     'heating_fuel_coal_coke_rate',
     'heating_fuel_solar_rate',
     'heating_fuel_other_rate',
     'heating_fuel_none_rate']
    
    # drop the rates columns
    new_df = df.drop(redundant, axis = 1)
    
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


    