import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from dash_apps import ingredient_utils as iu
import warnings


def create_sparse_matrix(matrix, grain_reference_list, hop_reference_list, yeast_reference_list): #assumes dense_matrix has ingredients as columns and time on rows

    current_matrix = matrix.transpose()
    # Convert time column names to integers
    current_matrix.columns = [int(col) for col in current_matrix.columns]
    # Calculate the max Time for this recipe
    last_ingredient_add = max([int(mat) for mat in current_matrix]) + 1

    # limit length of matrix to be 250 (ignores dry hops!!)
    if last_ingredient_add > 250:
        last_ingredient_add = 250
    # print('Last Ingredient Added (min): {0}'.format(last_ingredient_add))

    # Configure an expanded empty array with all zeros
    empty_array = np.zeros((current_matrix.shape[0], last_ingredient_add))
    padded_matrix = pd.DataFrame(empty_array.copy(), index=current_matrix.index)
    print('Original Shape: %s' %str(current_matrix.shape))
    print('New Shape: %s' %str(padded_matrix.shape))

    # Add in the type of ingredient
    # Add all ingredients from dense matrix to sparse matrix

    for i, index in enumerate(current_matrix.index):
        for j, column in enumerate(current_matrix.columns):
            padded_matrix.loc[index,column] = current_matrix.loc[index,column]
            try:
                if str(index) in grain_reference_list:
                    padded_matrix.loc[index, ['type']] = 'grain'
                elif str(index) in hop_reference_list:
                    padded_matrix.loc[index, 'type'] = 'hop'
                elif str(index) in yeast_reference_list:
                    padded_matrix.loc[index, 'type'] = 'yeast'
                elif str(index) == 'Temperature':
                    padded_matrix.loc[index, 'type'] = 'temperature'
                else:
                    padded_matrix.loc[index, 'type'] = 'other'
            except:
                continue

    padded_matrix.index.name='ingredients'
    temperature_sparse = padded_matrix.loc[padded_matrix.index == 'Temperature', :]

    return pd.DataFrame(padded_matrix), pd.DataFrame(temperature_sparse)


def create_tidy_matrix(sparse_matrix):

    tidy = pd.melt(sparse_matrix.reset_index(),
                   id_vars=['ingredients', 'type'], value_vars=list(range(0,len(sparse_matrix.columns))),
                   value_name='amount')

    # tidy.dropna(axis=0,inplace=True)
    tidy = tidy.loc[tidy['amount'] != 0, :]
    temperature_export = tidy.loc[tidy['ingredients'] == 'Temperature', :]
    tidy = tidy.loc[tidy['ingredients'] != 'Temperature', :]
    tidy['amount'] = tidy['amount'].apply(float)

    return tidy


def clean_tidy_and_export_temperature(temp_sparse_df_original):

    temp_sparse_df = temp_sparse_df_original.copy(deep=True)

    # If the first temperature is 0 (ie. didn't exist at time step 0) set the default to be room temp
    if temp_sparse_df.loc['Temperature',0] == 0.0:
        temp_sparse_df.loc['Temperature',0] = 22.0

    # If Temp == 0, roll through previous time steps until we find a valid temperature, carry that forward
    for i, each in enumerate(temp_sparse_df.columns):
        if temp_sparse_df.loc['Temperature',each] == 0.0:
            j = 1
            while j >= 0 and temp_sparse_df.loc['Temperature',each] == 0.0:
                temp_sparse_df.loc['Temperature',each] = temp_sparse_df.loc['Temperature',each-j]
                j += 1
        else:
            continue

    # Tidy
    tidy2 = pd.melt(temp_sparse_df.reset_index(),
                   id_vars=['ingredients', 'type'], value_vars=list(range(0,len(temp_sparse_df.columns))),
                   value_name='amount')

    # tidy2.dropna(axis=0,inplace=True)
    tidy2 = tidy2.loc[tidy2['amount'] != 0, :]
    temperature_export = tidy2.loc[tidy2['ingredients'] == 'Temperature', :]
    temperature_export['amount'] = temperature_export['amount'].apply(float)

    return temperature_export


def clean_brewers_friend(brewers):


    #################################
    ### abv
    # remove percent and convert to float
    def clean_abv(row):
        try:
            if row['abv'].lower()[-1:] == '%':
                return float(row['abv'][:-1])
            else:
                return None
        except:
            return row['abv']
    brewers['abv_clean'] = brewers.apply(clean_abv, axis=1)
    median = float(brewers['abv_clean'].median())
    brewers['abv_clean'].fillna(median, inplace=True)
    brewers.drop('abv', axis=1, inplace=True)


    #################################
    ### batch_size and batch_size_mode

    # Create Variable: Batch Size for Fermenter
    brewers.ix[brewers['batch_size_mode'] =='f', 'batch_size']
    brewers.ix[brewers['batch_size_mode'] =='f','batch_size_f'] = brewers.ix[brewers['batch_size_mode'] =='f', 'batch_size']
    brewers.ix[brewers['batch_size_mode'] !='f','batch_size_f'] = 0
    # Create Variable: Batch Size for Kettle
    brewers.ix[brewers['batch_size_mode'] =='k', 'batch_size']
    brewers.ix[brewers['batch_size_mode'] =='k','batch_size_k'] = brewers.ix[brewers['batch_size_mode'] =='k', 'batch_size']
    brewers.ix[brewers['batch_size_mode'] !='k','batch_size_k'] = 0
    # Drop Batch Size
    brewers.drop('batch_size', axis=1, inplace=True)
    # Drop Batch Size Mode
    brewers.drop('batch_size_mode', axis=1, inplace=True)


    #################################
    ### carbonation
    brewers['carbonation'].fillna(0, inplace=True)


    #################################
    ### boil_gravity
    # convert strings to floats, impute missing values using a median approach
    # also need to remove specific gravities above 2 (likely wrong units)
    def clean_boil_gravity(row):
        if row['boil gravity'] == 'n/a':
            return None
        if float(row['boil gravity']) > 2:
            return None
        return float(row['boil gravity'])
    brewers['boil_gravity_clean'] = brewers.apply(clean_boil_gravity, axis=1)
    median = float(brewers['boil_gravity_clean'].median())
    brewers['boil_gravity_clean'].fillna(median, inplace=True)
    brewers.drop('boil gravity', axis=1, inplace=True)
    # Cleaned and Ready to Use as boil_gravity_clean


    #################################
    ### boil_size
    # convert from gal's to L's, median impute for NaN's
    def clean_boil_size(row):
        gal_to_L = 3.7854
        try:
            if row['boil size'].lower()[-4:] == ' gal':
                return float(row['boil size'][:-4]) * gal_to_L
            if row['boil size'].lower()[-2:] == ' l':
                return float(row['boil size'][:-2])
            else:
                return None
        except:
            return row['boil size']
    brewers['boil_size_clean'] = brewers.apply(clean_boil_size, axis=1)
    median = float(brewers['boil_size_clean'].median())
    brewers['boil_size_clean'].fillna(median, inplace=True)
    brewers.drop('boil size', axis=1, inplace=True)


    #################################
    ### carbonation
    brewers['carbonation'].fillna(0, inplace=True)


    #################################
    ### color
    def clean_color(row):
        try:
            if row['color'].lower()[-2:] == '°l':
                return float(row['color'][:-2])
            else:
                return None
        except:
            return row['color']
    brewers['color_clean'] = brewers.apply(clean_color, axis=1)
    median = float(brewers['color_clean'].median())
    brewers['color_clean'].fillna(median, inplace=True)
    brewers.drop('color', axis=1, inplace=True)


    #################################
    ### efficiency_x
    def clean_efficiency_x(row):
        try:
            return float(row['efficiency_x'])
        except:
            return None
    brewers['efficiency_x_clean'] = brewers.apply(clean_efficiency_x, axis=1)
    median = float(brewers['efficiency_x_clean'].median())
    brewers['efficiency_x_clean'].fillna(median, inplace=True)
    brewers.drop('efficiency_x', axis=1, inplace=True)


    #################################
    ### fg_x
    # Removed any gravities above 1.5!
    def clean_fg_x(row):
        if float(row['fg_x']) < 1.5:
            return float(row['fg_x'])
        else:
            return None

    brewers['fg_clean'] = brewers.apply(clean_fg_x, axis=1)
    median = float(brewers['fg_clean'].median())
    brewers['fg_clean'].fillna(median, inplace=True)
    brewers.drop('fg_x', axis=1, inplace=True)


    #################################
    ### no_chill_extra_minutes
    # Default n/a's to 0
    def clean_no_chill(row):
        if row['no_chill_extra_minutes'] == 'n/a':
            return 0.0
        else:
            return float(row['no_chill_extra_minutes'])
    brewers['no_chill_clean'] = brewers.apply(clean_no_chill, axis=1)
    brewers['no_chill_clean'].fillna(0.0, inplace=True)
    brewers.drop('no_chill_extra_minutes', axis=1, inplace=True)

    #################################
    ### og_y
    # Removed any gravities above 1.5!
    def clean_og_y(row):
        if float(row['og_y']) < 1.5:
            return float(row['og_y'])
        else:
            return None

    brewers['og_clean'] = brewers.apply(clean_og_y, axis=1)
    median = float(brewers['og_clean'].median())
    brewers['og_clean'].fillna(median, inplace=True)
    brewers.drop('og_y', axis=1, inplace=True)

    #################################
    ### pitch_rate
    # Default n/a's to 0
    def clean_pitch_rate(row):
        if row['pitch rate'] == 'n/a':
            return 0.0
        else:
            return float(row['pitch rate'])
    brewers['pitch_rate_clean'] = brewers.apply(clean_pitch_rate, axis=1)
    brewers['pitch_rate_clean'].fillna(0.0, inplace=True)
    brewers.drop('pitch rate', axis=1, inplace=True)

    #################################
    ### primary_temp (use this one, no conversion from F to C necessary)
    def clean_primary_temp2(row):
        C_to_F = 1.8
        try:
            return float(row['primary_temp'])
        except:
            return None

    brewers['primary_temp_clean'] = brewers.apply(clean_primary_temp2, axis=1)
    median = float(brewers['primary_temp_clean'].median())
    brewers['primary_temp_clean'].fillna(median, inplace=True)
    brewers.drop('primary_temp', axis=1, inplace=True)


    #################################
    ### starting_mash_thickness
    # Default n/a's to 0
    def clean_starting_mash_thickness(row):
        if row['starting_mash_thickness'] == 'n/a':
            return 0.0
        else:
            return float(row['starting_mash_thickness'])
    brewers['starting_mash_thickness_clean'] = brewers.apply(clean_starting_mash_thickness, axis=1)
    brewers['starting_mash_thickness_clean'].fillna(0.0, inplace=True)
    brewers.drop('starting_mash_thickness', axis=1, inplace=True)


    #################################
    ### yield
    # convert from gal's to L's, median impute for NaN's
    def clean_yield(row):
        gal_to_L = 3.7854
        try:
            if row['yield'].lower()[-4:] == ' gal':
                return float(row['yield'][:-4]) * gal_to_L
            if row['yield'].lower()[-2:] == ' l':
                return float(row['yield'][:-2])
            else:
                return None
        except:
            return row['yield']

    brewers['yield_clean'] = brewers.apply(clean_yield, axis=1)
    median = float(brewers['yield_clean'].median())
    brewers['yield_clean'].fillna(median, inplace=True)
    brewers.drop('yield', axis=1, inplace=True)

    return brewers


def clean_export_results(recipes_df_cleaned, index):

    result_columns = ['boil time', 'brews', 'carbonation',
       'ibu_method', 'ibu_x', 'ibu_y', 'index',
       'name',
       'title', 'type',
       'untappd_rating', 'abv_clean',
       'batch_size_f', 'batch_size_k',
       'color_clean', 'efficiency_x_clean', 'fg_clean',
       'og_clean',
       'yield_clean', 'style_x', 'author']

    recipe_results = recipes_df_cleaned.copy(deep=True)
    recipe_results = recipe_results.loc[[index],result_columns].copy(deep=True)
    recipe_results.reset_index(inplace=True)

    return pd.DataFrame(recipe_results)


def clean_export_other_info(recipes_df_cleaned, index):

    other_recipe_input_columns = ['boil time', 'brews', 'carbonation',
       'method', 'miscs',
       'yeast_starter',
       'boil_gravity_clean', 'boil_size_clean', 'no_chill_clean', 'pitch_rate_clean', 'primary_temp_clean',
       'starting_mash_thickness_clean']

    recipe_other_info = recipes_df_cleaned.copy(deep=True)
    recipe_other_info = recipe_other_info.loc[[index],other_recipe_input_columns].copy(deep=True)
    recipe_other_info.reset_index(inplace=True)

    return pd.DataFrame(recipe_other_info)


def clean_and_export(df, index, grain_reference_list, hop_reference_list, yeast_reference_list):
    mat = pd.DataFrame(df.loc[index,'matrix'])
    sparse_matrix1, temperature_sparse1 = create_sparse_matrix(mat, grain_reference_list, hop_reference_list, yeast_reference_list)
    tidy_matrix1 = create_tidy_matrix(sparse_matrix1)
    temperature_export = clean_tidy_and_export_temperature(temperature_sparse1)
    recipe_results = clean_export_results(df, index)
    recipe_other_info = clean_export_other_info(df, index)

    return tidy_matrix1, temperature_export, recipe_results, recipe_other_info
