#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from nltk.corpus import stopwords
import os, zipfile, sys
from fuzzywuzzy import fuzz, process

"""
Unit conversions
"""

# Mass
def kg2oz(kg):
    return kg * 35.274
def kg2lb(kg):
    return kg * 2.20462

# Volumes
def l2gal(liters):
    return liters * .264172

# Colors
def ebc2srm(ebc):
    return ebc * .508
def lovibond2srm(lovibond):
    return (1.3546 * lovibond) - .76
def srm2ebc(srm):
    return srm * 1.97
def srm2lovibond(srm):
    return (srm + .76) / 1.3546

"""
Recipe feature extractors / formulas
"""

def get_ibu(recipe):   
    # We need to interpolate here because the table is only sampled data
    utilizations = utilization_table.drop('Time', axis=1)
    gravities = np.array(utilization_table.columns[1:], dtype=np.float)
    times = utilization_table['Time']
    interpolate_utilization = interp2d(gravities, times,  utilizations)

    hops = pd.DataFrame(recipe['hops'])
    
    volume, units = recipe['yield'].split()
    volume = float(volume)
    if units == 'l':
        volume = l2gal(volume)
    elif units == 'gal':
        pass
    else:
        print('Unknown yield units.')
    
    boil_gravity = recipe['boil gravity']
    if boil_gravity == 'n/a':
        return None
    boil_gravity = float(boil_gravity)
    total_ibu = 0
    
    for i, row in hops.iterrows():
        if 'Boil' in row['use']:
            alpha = float(row['alpha'])
            mass = kg2oz(float(row['amount']))
            time = int(row['time'])
            utilization = interpolate_utilization(boil_gravity, time)[0]
            aau = mass * alpha
            ibu = aau * utilization * 75 / volume
            total_ibu += ibu
#             print('{0} contributes {1} IBUs'.format(row['name'], ibu))
#     print('Total IBU: {0}'.format(total_ibu))
    return total_ibu

def get_SRM(recipe):
    volume_in_gallons = l2gal(recipe['batch_size'])
    grain_bill = pd.DataFrame(recipe['fermentables'])
    total_mcu = 0
    for i, row in grain_bill.iterrows():
        if row['add_after_boil'] == 'false' or row['add_after_boil'] == 'False':
            grain_color = np.float(row['color'])
            grain_weight_in_lbs = kg2lb(np.float(row['amount']))
            mcu =  grain_weight_in_lbs * grain_color / volume_in_gallons
            total_mcu += mcu
#             print("{0} provides {1} SRM.".format(row['name'], srm))
    srm = 1.4922 * (total_mcu**0.6859)
    return srm

def get_OG(recipe):
    # TODO
    pass

def get_FG(recipe):
    og = get_OG(recipe)
    att = recipe['yeasts']['attenuation']
    return -att * (og-1) + og

def get_ABV(recipe):
    return (get_OG(recipe) - get_FG(recipe)) * 131.25

def get_balance(recipe):
    og = get_OG(recipe)
    ibu = get_ibu(recipe)
    points = (og - 1) * 1000
    return ibu / points

"""
Data loading
"""

def load_all_subsets():
    dfs = []
    search_dir = 'data/subsets/'
    filenames = os.listdir(search_dir)
    for filename in filenames:
        dfs.append(pd.read_json(search_dir + filename))
        if len(dfs) % 1000 == 0:
            print("Loaded {0} subsets ({1:.1f}%).".format(len(dfs), len(dfs) / len(filenames) * 100))
    recipes = pd.concat(dfs).reset_index(drop=True)
    print("Loaded all {0} subsets with {1} recipes.".format(len(dfs), len(recipes)))
    return recipes

def load_all_recipes(json_file='data/all_recipes.json', zip_file='all_recipes.zip', destination='.'):
    full_path = '/'.join([destination, json_file])
    if not os.path.isfile(full_path):
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()
    recipes = pd.read_json(full_path)
    print("Loaded {0} recipes.".format(len(recipes)))
    return recipes

"""
Ingredient functions
"""

def get_all_ingredients(recipes):
    """
    Get lists of all fermentables, grains, and yeasts within the recipe set.
    Please note that this list is kind of strange, because it's all of the hop
    additions but without whole duplicates. So there will be duplicate names of
    ingredients if they have differing other properties like alpha acids.
    """
    
    fermentable_columns = ['color', 'name', 'origin', 'type', 'version', 'yield']
    hop_columns = ['alpha', 'form', 'name', 'version']
    yeast_columns = ['attenuation', 'flocculation', 'form', 'laboratory', 'max_temperature', 'min_temperature', 'name', 'product_id', 'type', 'version']
    
    all_fermentables = pd.concat([pd.DataFrame(recipe['fermentables']) for i,recipe in recipes.iterrows()])[fermentable_columns].drop_duplicates().reset_index(drop=True)
    all_hops = pd.concat([pd.DataFrame(recipe['hops']) for i,recipe in recipes.iterrows()])[hop_columns].drop_duplicates().reset_index(drop=True)
    all_yeasts = pd.concat([pd.DataFrame(recipe['yeasts']) for i,recipe in recipes.iterrows()])[yeast_columns].drop_duplicates().reset_index(drop=True)
    print("Ingredient counts: {0} fermentables, {1} hops, and {2} yeasts".format(len(all_fermentables), len(all_hops), len(all_yeasts)))
    return all_fermentables, all_hops, all_yeasts

def recipe_ingredients(recipe):
    fermentable_columns = ['color', 'name', 'origin', 'type', 'version', 'yield']
    hop_columns = ['alpha', 'form', 'name', 'version']
    yeast_columns = ['attenuation', 'flocculation', 'form', 'laboratory', 'max_temperature', 'min_temperature', 'name', 'product_id', 'type', 'version']
#     print(recipe['yeasts'])
#     all_fermentables = pd.concat([pd.DataFrame(recipe['fermentables']) for i,recipe in recipes.iterrows()])[fermentable_columns].drop_duplicates().reset_index(drop=True)
#     all_hops = pd.concat([pd.DataFrame(recipe['hops']) for i,recipe in recipes.iterrows()])[hop_columns].drop_duplicates().reset_index(drop=True)
#     all_yeasts = pd.concat([pd.DataFrame(recipe['yeasts']) for i,recipe in recipes.iterrows()])[yeast_columns].drop_duplicates().reset_index(drop=True)
    all_fermentables = pd.DataFrame(recipe['fermentables'])[fermentable_columns]
    all_hops = pd.DataFrame(recipe['hops'])[hop_columns]
    all_yeasts = pd.DataFrame(recipe['yeasts'])[yeast_columns]
    all_ingredients = pd.concat([all_fermentables['name'], all_hops['name'], all_yeasts['name']]).reset_index(drop=True)
    return all_ingredients

def recipe_to_matrix(recipe, ingredients_list):
    recipe_ingredients = recipe_ingredients(recipe)
    
def recipe_to_steps_old(recipe):
    import time
    times=[]
    times.append(time.time())
    times.append(time.time())
    print("A: {0}".format(times[-1] - times[-2]))
    bills = []
    ferm_bill = pd.DataFrame(recipe['fermentables'])
    hop_bill = pd.DataFrame(recipe['hops'])
    misc_bill = pd.DataFrame(recipe['miscs'])
    yeast_bill = pd.DataFrame(recipe['yeasts'])
    yeast_bill['time'] = ''
    times.append(time.time())
    print("B: {0}".format(times[-1] - times[-2]))
    mash_time = 60
    average_yeast_temp = recipe.primary_temp
    max_mash_temp = 67
    if('min_temperature' in yeast_bill.columns):
        average_yeast_temp = np.array([yeast_bill['min_temperature'].astype(np.float), yeast_bill['max_temperature'].astype(np.float)]).mean()
    times.append(time.time())
    print("C: {0}".format(times[-1] - times[-2]))
    if len(ferm_bill) > 0:
        ferm_names = (ferm_bill['origin'].fillna('') + ' '+ pd.DataFrame(recipe['fermentables'])['name']).str.strip().rename('name').reset_index(drop=True)
        ferm_amounts = ferm_bill['amount'].reset_index(drop=True)
        
        
        # Offset for fermentables that occur after the boil, like sugar additions
        mash_steps = recipe['mash']['mash_steps']
        if mash_steps['0'] is not None:
            mash_time = sum([float(x) if x is not None else 0.0 for x in pd.DataFrame(mash_steps).transpose()['step_time'].iloc[0].values()])
            max_mash_temp = np.array([float(x) for x in pd.DataFrame(mash_steps).transpose()['step_temp'].iloc[0].values()]).mean()
        timing_offset = [0 if use  == "false" else (mash_time + recipe['boil_time']) for use in ferm_bill['add_after_boil']]
        
        ferm_times = pd.Series(timing_offset).rename('time').reset_index(drop=True)
        ferm_form = ferm_bill['type'].rename('form').reset_index(drop=True)
        ferm_type = pd.Series(['Fermentable']*len(ferm_bill)).rename('type').reset_index(drop=True)
        # For now just set it to the max mash temp. This removes and infusion steps, but that's fine for an MVP.
        temps = [max_mash_temp if use  == "false" else recipe.primary_temp for use in ferm_bill['add_after_boil']]
        ferm_temps = pd.Series(temps, name='temperature')
        ferm_bill = pd.concat([ferm_names, ferm_amounts, ferm_times, ferm_form, ferm_type, ferm_temps], axis=1)
        bills.append(ferm_bill)
    times.append(time.time())
    print("Ferm bill: {0}".format(times[-1] - times[-2]))
    if len(hop_bill) > 0:
        boil_or_ferm = [-1 if use  == "Boil" else 1 for use in hop_bill["use"]]
        hop_times_from_mash = recipe['boil_time']  + boil_or_ferm * hop_bill['time'].astype(np.float)
        
    
        temps = [100 if use  == "Boil" else recipe.primary_temp for use in hop_bill["use"]]

        hop_times_absolute = hop_times_from_mash + mash_time
        hop_times_absolute
        hop_bill = hop_bill[['name', 'amount', 'time', 'form']]
        hop_bill['type'] = ['Hop'] * len(hop_bill)
        hop_bill['time'] = hop_times_absolute
        hop_bill['temperature'] = temps
        hop_bill['temperature'].fillna(average_yeast_temp, inplace = True)
        bills.append(hop_bill)
    times.append(time.time())
    print("Hop bill: {0}".format(times[-1] - times[-2]))
    if len(misc_bill) > 0:
        times.append(time.time())
        print("time: {0}".format(times[-1] - times[-2]))
        misc_bill['time'] = misc_bill['time'].astype(np.float)
        
        temps = [100 if use  == "Boil" else max_mash_temp if use == 'Mash' else recipe.primary_temp for use in misc_bill['use']]
        times_absolute = [mash_time + recipe['boil_time'] - row['time'] if row['use']  == "Boil" else mash_time - row['time'] if row['use'] == 'Mash' else mash_time + recipe['boil_time'] + row['time'] for i,row in misc_bill.iterrows()]

        misc_bill = misc_bill[['name', 'amount', 'time', 'type']]
        times.append(time.time())
        print("else: {0}".format(times[-1] - times[-2]))
        misc_bill['time'] = times_absolute
        times.append(time.time())
        print("time abs: {0}".format(times[-1] - times[-2]))
        misc_bill['temperature'] = temps
        times.append(time.time())
        print("temp: {0}".format(times[-1] - times[-2]))
        bills.append(misc_bill)
    
    times.append(time.time())
    print("Misc Bill: {0}".format(times[-1] - times[-2]))
    if len(yeast_bill) > 0:
        times.append(time.time())
        print("starting: {0}".format(times[-1] - times[-2]))
        yeast_bill = yeast_bill[['name', 'amount', 'form', 'type']]
        times.append(time.time())
        print("filtering yeast: {0}".format(times[-1] - times[-2]))
        #print(yeast_bill['time'])
        #yeast_bill['time'] = mash_time + recipe['boil_time']
        yeast_bill.assign(time=mash_time + recipe['boil_time'])
        #print(yeast_bill['time'])
        times.append(time.time())
        print("yeast time: {0}".format(times[-1] - times[-2]))
        yeast_bill['type'] = 'Yeast'
        times.append(time.time())
        print("yeast type: {0}".format(times[-1] - times[-2]))
        yeast_bill['temperature'] = recipe.primary_temp        
        times.append(time.time())
        print("yeast temp: {0}".format(times[-1] - times[-2]))
        yeast_bill['temperature'].fillna(average_yeast_temp, inplace = True)
        times.append(time.time())
        print("fillna: {0}".format(times[-1] - times[-2]))
        bills.append(yeast_bill)
    times.append(time.time())
    print("Yeast bill: {0}".format(times[-1] - times[-2]))
    steps = pd.concat(bills)
    times.append(time.time())
    print("Concat: {0}".format(times[-1] - times[-2]))
    steps['time'] = steps['time'].fillna(0)
    steps = steps.sort_values(by='time').reset_index(drop=True)
    times.append(time.time())
    print("Sort: {0}".format(times[-1] - times[-2]))
    return steps

def recipe_to_steps(recipe):
    bills = []
    ferm_bill = pd.DataFrame(recipe['fermentables'])
    hop_bill = pd.DataFrame(recipe['hops'])
    misc_bill = pd.DataFrame(recipe['miscs'])
    yeast_bill = pd.DataFrame(recipe['yeasts'])
    yeast_bill['time'] = ''
    mash_time = 60
    average_yeast_temp = recipe.primary_temp
    max_mash_temp = 67
    if('min_temperature' in yeast_bill.columns):
        average_yeast_temp = np.array([yeast_bill['min_temperature'].astype(np.float), yeast_bill['max_temperature'].astype(np.float)]).mean()
    if len(ferm_bill) > 0:
        ferm_names = (ferm_bill['origin'].fillna('') + ' '+ pd.DataFrame(recipe['fermentables'])['name']).str.strip().rename('name').reset_index(drop=True)
        ferm_amounts = ferm_bill['amount'].reset_index(drop=True)
               
        # Offset for fermentables that occur after the boil, like sugar additions
        mash_steps = recipe['mash']['mash_steps']
        if mash_steps['0'] is not None:
            mash_time = sum([float(x) if x is not None else 0.0 for x in pd.DataFrame(mash_steps).transpose()['step_time'].iloc[0].values()])
            max_mash_temp = np.array([float(x) for x in pd.DataFrame(mash_steps).transpose()['step_temp'].iloc[0].values()]).mean()
        timing_offset = [0 if use  == "false" else (mash_time + recipe['boil_time']) for use in ferm_bill['add_after_boil']]
        
        ferm_times = pd.Series(timing_offset).rename('time').reset_index(drop=True)
        ferm_form = ferm_bill['type'].rename('form').reset_index(drop=True)
        ferm_type = pd.Series(['Fermentable']*len(ferm_bill)).rename('type').reset_index(drop=True)
        # For now just set it to the max mash temp. This removes and infusion steps, but that's fine for an MVP.
        temps = [max_mash_temp if use  == "false" else recipe.primary_temp for use in ferm_bill['add_after_boil']]
        ferm_temps = pd.Series(temps, name='temperature')
        ferm_bill = pd.concat([ferm_names, ferm_amounts, ferm_times, ferm_form, ferm_type, ferm_temps], axis=1)
        bills.append(ferm_bill)
    if len(hop_bill) > 0:
        boil_or_ferm = [-1 if use  == "Boil" else 1 for use in hop_bill["use"]]
        hop_times_from_mash = recipe['boil_time']  + boil_or_ferm * hop_bill['time'].astype(np.float)
        
    
        temps = [100 if use  == "Boil" else recipe.primary_temp for use in hop_bill["use"]]

        hop_times_absolute = hop_times_from_mash + mash_time
        hop_bill = hop_bill.filter(['name', 'amount', 'form', 'type'], axis=1)

        hop_bill['type'] = ['Hop'] * len(hop_bill)
        hop_bill['time'] = hop_times_absolute
        hop_bill['temperature'] = temps
        hop_bill['temperature'].fillna(average_yeast_temp, inplace = True)
        bills.append(hop_bill)
    if len(misc_bill) > 0:
        misc_bill['time'] = misc_bill['time'].astype(np.float)
        
        temps = [100 if use  == "Boil" else max_mash_temp if use == 'Mash' else recipe.primary_temp for use in misc_bill['use']]
        times_absolute = [mash_time + recipe['boil_time'] - row['time'] if row['use']  == "Boil" else mash_time - row['time'] if row['use'] == 'Mash' else mash_time + recipe['boil_time'] + row['time'] for i,row in misc_bill.iterrows()]

        misc_bill = misc_bill.filter(['name', 'amount', 'form', 'type'], axis=1)
        misc_bill['time'] = times_absolute
        misc_bill['temperature'] = temps
        bills.append(misc_bill)
    
    if len(yeast_bill) > 0:
        yeast_bill = yeast_bill.filter(['name', 'amount', 'form', 'type'], axis=1)
        yeast_bill['time'] = mash_time + recipe['boil_time']
        yeast_bill['type'] = 'Yeast'
        yeast_bill['temperature'] = recipe.primary_temp        
        yeast_bill['temperature'].fillna(average_yeast_temp, inplace = True)
        bills.append(yeast_bill)
    steps = pd.DataFrame(columns=['name', 'amount', 'form', 'type'])
    if len(bills) > 0:
        steps = pd.concat(bills)
        steps['time'] = steps['time'].fillna(0)
        steps = steps.sort_values(by='time').reset_index(drop=True)
    return steps

def recipes_to_matrices_np(recs, max_time=21600, sparse=False, include_temperature=True, stack=False, clean=True, cleaned_ref_names=None, cleaned_ref_aliases=None, bin_unk=False):
    #all_steps = [recipe_to_steps(recipe) for i,recipe in recipes.iterrows()]
    all_steps = []
    recipes = recs.copy().reset_index(drop=True).sort_index()
#    print(recipes.iloc[0].steps)
#    print('recipes: ' + str(len(recipes)))
    recipes['matrix'] = ''
    #f 'matrix' not in recipes.columns:
    #   recipes['matrix'] = ''
    # If we've precalculated the steps, we'll use those to save time.
    if len(recipes) > 0 and 'steps' in recipes.columns:
        print('Using precalculated steps.')
        all_steps = [pd.DataFrame(recipe.steps) for i,recipe in recipes.iterrows()]
    else:
        print('Calculating steps.')
        all_steps = [recipe_to_steps(recipe) for i,recipe in recipes.iterrows()]
    if sys.version_info[0] > 2:
        all_steps = list(filter((None).__ne__, all_steps))
    else:
        all_steps = list(filter(lambda x: x is not None, all_steps))
    if clean:
        
        for steps in all_steps:
            steps['name'], _ = clean_ingredients(steps['name'])
            
            # Right now both need to be handed in. That's not necessarily good though, but we're kind of building this with a small team, so this should be fine.
            if cleaned_ref_names is not None and cleaned_ref_aliases is not None:
                steps['name'] = replace_aliases(steps['name'], cleaned_ref_names, cleaned_ref_aliases, fuzzy=True, bin_unk=bin_unk)
    # We need everything in a single DF in order to get the max time and a list of all ingredients. This is kinda crummy and should perhaps be rewritten.
    all_steps_single_df = pd.concat(all_steps)
    columns = all_steps_single_df['name'].drop_duplicates().reset_index(drop=True)
    max_time = int(all_steps_single_df['time'].max())
    if include_temperature:
        columns = columns.append(pd.Series('Temperature'))
    matrices = []
    
    minutes = range(0, max_time+1)
    #recipes['matrix'] = np.array([])
    for j, steps in enumerate(all_steps):

        #This will allow us to only index the times we care about.
        mat = pd.DataFrame(columns=columns, dtype=np.float)
        if sparse:
            # This creates a very sparse matrix
            mat = pd.DataFrame(index=minutes, columns=columns, dtype=np.float)
        for i,row in steps.iterrows():
            ingredient_name = row['name']
            time = row['time']
            mat.at[int(time), ingredient_name] = row['amount']
            #mat.iloc[int(time)][ingredient_name] = row['amount']
            if include_temperature:
                #mat.set_value(int(time), 'Temperature', row['temperature'])
                mat.at[int(time), 'Temperature'] = row['temperature']
                #mat.iloc[int(time)]['Temperature'] = row['temperature']
        mat.index.name='time'
        #if drop_empty_minutes:
        #    mat = mat.dropna(how='all')
        mat.fillna(0, inplace=True)
        #matrices.append(mat)
        recipes.set_value(j, 'matrix', mat)
    if stack:
        return np.stack(matrices)
    #return matrices
    return recipes


def recipes_to_matrices(recs, max_time=21600, drop_empty_minutes=True, include_temperature=True, stack=False, clean=True, cleaned_ref_names=None, cleaned_ref_aliases=None, bin_unk=False):
    #all_steps = [recipe_to_steps(recipe) for i,recipe in recipes.iterrows()]
    all_steps = []
    recipes = recs.copy()
    #f 'matrix' not in recipes.columns:
    #   recipes['matrix'] = ''
    # If we've precalculated the steps, we'll use those to save time.
    if len(recipes) > 0 and 'steps' in recipes.columns:
        print('Using precalculated steps.')
        all_steps = [pd.DataFrame(recipe.steps) for i,recipe in recipes.iterrows()]
    else:
        print('Calculating steps.')
        all_steps = [recipe_to_steps(recipe) for i,recipe in recipes.iterrows()]
    if sys.version_info[0] > 2:
        all_steps = list(filter((None).__ne__, all_steps))
    else:
        all_steps = list(filter(lambda x: x is not None, all_steps))
    if clean:
        
        for steps in all_steps:
            steps['name'], _ = clean_ingredients(steps['name'])
            
            # Right now both need to be handed in. That's not necessarily good though, but we're kind of building this with a small team, so this should be fine.
        if cleaned_ref_names is not None and cleaned_ref_aliases is not None:
            steps['name'] = replace_aliases(steps['name'], cleaned_ref_names, cleaned_ref_aliases, fuzzy=True, bin_unk=bin_unk)
    # We need everything in a single DF in order to get the max time and a list of all ingredients. This is kinda crummy and should perhaps be rewritten.
    all_steps_single_df = pd.concat(all_steps)
    columns = all_steps_single_df['name'].drop_duplicates().reset_index(drop=True)
    max_time = int(all_steps_single_df['time'].max())
    if include_temperature:
        columns = columns.append(pd.Series('Temperature'))
    matrices = []
    
    minutes = range(0, max_time+1)
    for i, steps in enumerate(all_steps):
        mat = pd.DataFrame(index=minutes, columns=columns, dtype=np.float)
        for i,row in steps.iterrows():
            ingredient_name = row['name']
            time = row['time']
            mat.iloc[int(time)][ingredient_name] = row['amount']
            if include_temperature:
                mat.iloc[int(time)]['Temperature'] = row['temperature']
        mat.index.name='time'
        if drop_empty_minutes:
            mat = mat.dropna(how='all')
        mat.fillna(0, inplace=True)
        matrices.append(mat)
    if stack:
        return np.stack(matrices)
    return matrices

def get_series_sizes(series):
        return (len(np.unique(series)), len(series), len(np.unique(series)) / len(series))
    
def clean_ingredients(dirty, remove_commas=True, verbose=False):
    
    sizes = []
    sizes.append(get_series_sizes(dirty))
    if verbose:
        print("Initial size (unique/total = %): {0}/{1} = {2:.3f}%".format(*sizes[-1]))
    
    # Lower case everything
    clean = dirty.copy()
    clean = clean.str.lower()
    sizes.append(get_series_sizes(clean))
    if verbose:
        print("Lowered size (unique/total = %): {0}/{1} = {2:.3f}%".format(*sizes[-1]))

    # Remove stop words
    stop = stopwords.words('english')
    clean = clean.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    sizes.append(get_series_sizes(clean))
    if verbose:
        print("Stopwordless size (unique/total = %): {0}/{1} = {2:.3f}%".format(*sizes[-1]))

    # Strip whitespace (Redundant?)
    clean = clean.str.strip()
    sizes.append(get_series_sizes(clean))
    if verbose:
        print("Stripped size (unique/total = %): {0}/{1} = {2:.3f}%".format(*sizes[-1]))

    # Remove non alpha numerics   
    regex_str = r'([^\s\w]|_)+'
    if not remove_commas:
        regex_str = r'([^\s\w,]|_)+'
    
    clean = clean.str.replace(regex_str, '').str.replace(r'\s\s+', ' ')
    ## FIXME Rerun this cleaning on everything to make sure it actually strips out whitespace correctly.
    clean = clean.str.strip()
    sizes.append(get_series_sizes(clean))
    if verbose:
        print("Scrunched size (unique/total = %): {0}/{1} = {2:.3f}%".format(*sizes[-1]))
    return clean, sizes

def fuzzy_match(x, choices, scorer, cutoff):
    matches = process.extractOne(x, 
                            choices=choices, 
                            scorer=scorer, 
                            score_cutoff=cutoff)
    if matches is not None:
        return matches[0]
    return x

"""
def replace_alias_old(given_value, reference_ingredients, reference_aliases, full_match=True):
    real_names = []    
    if full_match:
        real_names = reference_ingredients[[given_value in row for row in reference_aliases.str.split(',')]]
    else:
        real_names = reference_ingredients[reference_aliases.str.contains(given_value)]
    if len(real_names) < 1:
        return given_value
    return real_names.iloc[0]
"""


def replace_alias(given_value, reference_ingredients, reference_aliases, full_match=True, bin_unk=False):
    real_names = []
    # FIXME rather than using smarter returns I set a choice variable so we can debug
    if pd.Series(reference_ingredients.str.strip() == given_value).sum() > 0:
        choice = given_value
    else:
        if full_match:
            real_names = reference_ingredients[[given_value in row for row in reference_aliases.str.split(',')]].copy()
        else:
            real_names = reference_ingredients[reference_aliases.str.contains(given_value)]
        if len(real_names) < 1:
            if bin_unk:
                choice = 'unk_ingredient'
            else:
                choice = given_value
        else:
            choice = real_names.iloc[0]
    #print(given_value, ' -> ' , choice)
    return choice

def replace_aliases(ingredient_list, reference_names, reference_aliases, fuzzy=True, bin_unk=False):
    ingredient_list_copy = ingredient_list.copy()
    if fuzzy:
        # FIXME Ideally this should probably be computed once rather than for every single cell.
        # Place all of the names and aliases together in a list
        potential_names = np.concatenate([np.hstack([x for x in reference_aliases.str.split(',')]), reference_names] )
        # Find the best fuzzy match within that list and set the value to be that
        ingredient_list_copy = ingredient_list_copy.apply(fuzzy_match, args=(potential_names, fuzz.token_sort_ratio, 80))
        
    return ingredient_list_copy.map(lambda x: replace_alias(x, reference_names, reference_aliases, bin_unk=bin_unk))


"""
Visualization and Reporting
"""

def plot_recipe_timeline(recipe):
    steps = recipe_to_steps(recipe)
    plt.figure(figsize=(14,14))
    plt.plot(steps["time"], steps["temperature"], '-o', drawstyle='steps-post')

    # # add annotations one by one with a loop
    # # beginning_of_groups = steps['temperature'].drop_duplicates().index
    offset = 0
    last_pos = [-1, -1]
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Temperature (°C )')
    plt.title('Brew Timeline')
    for line in range(0,steps.shape[0]):

        x = steps.time[line]+0.2
        y = steps.temperature[line]
        if([x,y]!=last_pos):
            offset=np.random.randint(2)
        y += offset
        plt.text(x, y, '({0}) {1}'.format(steps.amount[line], steps.name[line]), horizontalalignment='left', size='medium', color='black', weight='semibold')
        offset -= 5
        last_pos = [x,steps.temperature[line]]
    plt.show() 


def matrix_heatmap(matrix):
    plt.figure(figsize=(5,5))
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')
    ax.set(xlabel='Features', ylabel='Time')
    plt.show()

def heatmap_matrices(matrices):
    fig = plt.figure(figsize=(16,10))
    for i, matrix in enumerate(matrices):
        ax = fig.add_subplot(len(matrices) * 100 + 10 + (i+1))
        sns.heatmap(matrix, annot=True, cmap='Blues', ax=ax, vmin=0, vmax=100)
        ax.set(xlabel='Features', ylabel='Time')
    plt.show()