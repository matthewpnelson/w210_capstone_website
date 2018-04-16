# Standard python helper libraries.
import json
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


def load_commercial_beer_data():
    def load_beer_data(filename, file_format='json'):
        if file_format == 'json':
            beer_dict = json.load(open(filename+'.json'))
        elif file_format == 'pickle':
            pass
        else:
            print('Incorrect file format {}'.format(file_format))
        return beer_dict

    beers = pd.DataFrame(load_beer_data('mehdi_beer_data')).transpose()

    # Cleaning Script
    beers['rating'] = beers['rating'].astype(float)
    beers['raters'] = beers['raters'].astype(int)

    # Build a Lookup Name Column
    beers["lookup_name"] = beers['brewery'] + " " + beers['name']

    # Strip out the '\nShow Less\n' in all descriptions
    beers['description'] = beers['description'].map(lambda x: x.rstrip('\nShow Less\n'))

    ############ Could filter the beers that only have a few ratings. Won't worry about this for now.
    # Drop Beers with fewer than 5000 ratings
    # beers = beers.loc[beers['raters'] > 5000, :]

    # Drop Beers without an average rating
    beers.dropna(axis=0, subset=['rating'], inplace=True)
    beers.reset_index(inplace=True, drop=True)

    # Missing a lot of ibu data, median impute for each beer style for now. default to median for all beers
    median_ibu = float(beers['ibu'].median())
    style_ibus = {}

    for style in list(pd.unique(beers['style'])):
        current_style_median = beers.loc[beers['style'] == style, 'ibu'].median()

        if math.isnan(current_style_median):
            style_ibus[style] = median_ibu
        else:
            style_ibus[style] = current_style_median

    def impute_ibu(row):
            if math.isnan(row['ibu']):
                try:
                    return style_ibus[row['style']]
                except:
                    return median_ibu
            else:
                return row['ibu']

    beers['ibu'] = beers.apply(impute_ibu, axis=1)

    # Missing a few abv data, median impute for each beer style for now. default to median for all beers
    median_abv = float(beers['abv'].median())
    style_abvs = {}

    for style in list(pd.unique(beers['style'])):
        current_style_median = beers.loc[beers['style'] == style, 'abv'].median()

        if math.isnan(current_style_median):
            style_abvs[style] = median_abv
        else:
            style_abvs[style] = current_style_median

    def impute_abv(row):
            if math.isnan(row['abv']):
                try:
                    return style_abvs[row['style']]
                except:
                    return median_abv
            else:
                return row['abv']

    beers['abv'] = beers.apply(impute_abv, axis=1)

    # Encode the Beer Types as numerical
    le = LabelEncoder()
    le.fit(beers['style'])
    # le.classes_
    beers['style_enc'] = le.transform(beers['style'])

    # Export le Model
    joblib.dump(le, 'le_model.pkl')
    return beers

def create_tokens(description, tokenizer):
    # Adjust all words in sentence to Lowercase
    description = str(description.lower())
    #Tokenize the sentence
    tokens = tokenizer.tokenize(description)
    return tokens


def buildDescVector(google_model, tokens, vector_dim):
    # Initialize a blank vector of shape (1,vector_dim)
    desc_vec = np.zeros(vector_dim).reshape((1, vector_dim))
    # Initialize # of words in tweet count
    count = 0.
    for word in tokens:
        try:
            # Take each word vector in the tweet and add each dimension to the overall tweet vector
            desc_vec += google_model[word].reshape((1, vector_dim))
            # Increment Count
            count += 1.
        except:
            continue
    # Average each vector parameter by the # of words in the tweet (pass if no words exist in tweet)
    if count != 0:
        desc_vec /= count
    # Return Tweet Vector
    return desc_vec
