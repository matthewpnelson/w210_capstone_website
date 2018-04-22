import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np

# from dash_apps import ingredient_utils as iu
# from dash_apps import data_cleanup_viz as dcv
import warnings

## Dunno if I need this
from flask import Flask

# Standard python helper libraries.
import collections
import itertools
import json
import os
import re
import sys
import time
import math
import copy
import random

# Numerical manipulation libraries.
from scipy import stats
import scipy.optimize

#NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.

# Word2Vec Model
import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class

# Machine Learning Packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

# Helper functions
from dash_apps import beer_utilities

###############################
from run_server2 import app as server
app = dash.Dash(name='recommend_beer', sharing=True, server=server, url_base_pathname='/recommend_beer')
app.config.suppress_callback_exceptions = False
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'  # noqa: E501
    })


###### IMPORT DATA ##############

# Google Word2Vec Encoding Model
google_model = gensim.models.KeyedVectors.load_word2vec_format('/root/alegorithm_data/GoogleNews-vectors-negative300.bin', binary=True)

# Use NLTK's Tweet Tokenizer
t = TweetTokenizer()

# Load in Pre-cleaned & Adjusted Beers Data
beers = pd.read_json('/root/alegorithm_data/beers_data_py2.json')
# beers['abv'] = beers['abv'] *  100

# Load in Pretrained Label Encoder Model
le = joblib.load('/root/alegorithm_data/le_model_py2.pkl')

# Load in Pretrained KNN Model
knn_model = joblib.load('/root/alegorithm_data/knn_model_py2.pkl')

# Specify the Word Vector Dimensionality
vector_dim = 300 #matches the google model

# Grab Locations for Breweries
def add_brewery_info(beers):
    # breweries = pd.read_json('dash_apps/mehdi_brewery_data.json')
    breweries = pd.DataFrame(json.load(open('/root/alegorithm_data/mehdi_brewery_data.json')))
    beers = breweries.transpose().merge(beers, left_index=True, right_on='brewery', how='right', suffixes=['_x',''])
    beers['brewery_x'].fillna('blank', inplace=True)

    def getCity(address):
        if address == 'blank':
            return 'blank'
        else:
            address = address.split(',')
            return address[0]

    def getStateProvince(address):
        if address == 'blank':
            return 'blank'
        else:
            address = address.split(',')
            if 'United States' in address[-1]:
                return address[-1][1:3]
            elif 'Canada' in address[-1]:
                return address[-1][1:3]
            else:
                return 'blank'

    def getCountry(address):
        if address == 'blank':
            return 'blank'
        else:
            address = address.split(',')
            if 'United States' in address[-1]:
                return 'United States'
            else:
                last = address[-1].split(' ')
                return last[-1]

    def rowCity(row):
        return getCity(row['brewery_x'])
    def rowState(row):
        return getStateProvince(row['brewery_x'])
    def rowCountry(row):
        return getCountry(row['brewery_x'])

    beers['city'] = beers.apply(rowCity, axis=1)
    beers['state'] = beers.apply(rowState, axis=1)
    beers['country'] = beers.apply(rowCountry, axis=1)
    return beers

beers = add_brewery_info(beers)

############################################

# Create controls

beer_index_options = [{'label': str(beers.loc[i, 'lookup_name']), 'value': i} for i in beers.index]
beer_style_options = [{'label': str(i), 'value': i}
                  for i in beers['style'].unique()]
brewery_options = [{'label': str(i), 'value': i}
                  for i in beers['brewery'].unique()]

city_options = [{'label': str(i), 'value': i}
                  for i in beers['city'].unique()]
state_options = [{'label': str(i), 'value': i}
                  for i in beers['state'].unique()]
country_options = [{'label': str(i), 'value': i}
                  for i in beers['country'].unique()]

# Input from a User Profile (test)
test_beers = ['Speakeasy Ales & Lagers Blind Tiger Imperial IPA',
 '18th Street Brewery The Fox And the Goat',
 "B.O.B.'s Brewery Hoptics",
 '3 Floyds Brewing Company Yum Yum',
 'Commonhouse Aleworks Broad Path Brown',
 'The Kernel Brewery India Pale Ale Double Black',
 '3 Sheeps Brewing Company Bourbon Barrel Aged Paid Time Off',
 'Spearfish Brewing Elephant Man',
 'East Nashville Beer Works East Bank Citra IPA',
 'Cahaba Brewing Company English Oat Brown',
 'Tampa Beer Works Saint Bosco IPA',
 'Casa Agria Specialty Ales Pinky Sour',
 "Samuel Smith's Old Brewery India Ale",
 'Mad Squirrel Brewing Company London Porter',
 'The Alchemist Focal Banger',
 'O-Twn Brewing Das Panzer',
 'Ocean Lab Brewing Co. Hurricaned Harvest Ale 2017',
 "The Brewer's Art Resurrection",
 'Mad Frog Brewery Orange Dub Witbier',
 'Bad Habit Brewing Company Rather Ripped Series: Blood Orange Milkshake IPA']
test_ratings = [3.9100000000000001,
 3.8200000000000003,
 3.71,
 3.8100000000000001,
 3.6499999999999999,
 4.5099999999999998,
 4.3499999999999996,
 3.5,
 3.6899999999999999,
 4.3499999999999996,
 3.9100000000000001,
 4.0599999999999996,
 3.4300000000000002,
 3.71,
 4.4900000000000002,
 4.3099999999999996,
 4.0800000000000001,
 3.71,
 3.7599999999999998,
 4.0899999999999999]
user_profile_beer_list = dict(beer=test_beers, rating=test_ratings)

# Layout
layout = dict(
    autosize=True,
    height=600,
    font=dict(color='#333640'),
    titlefont=dict(color='#333640', size='14'),
    margin=dict(
        l=50,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    legend=dict(font=dict(size=10), orientation='h', x=0, y=0),
    title='',
    xaxis={'visible':False, 'type': 'linear', 'ticks': '', 'showticklabels': False},#, 'title': 'Brew Time (min)'},
    yaxis={'visible':False, 'type': 'linear', 'ticks': '', 'showticklabels': False},#, 'title': 'Ingredients (yeasts: oz, fermentables/hops: kg)'},

)

# In[]:
# Create app layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    'Beer Recommendation Assistant',
                                    className='eight columns',
                                ),
                                html.Img(
                                    src="https://s3-us-west-2.amazonaws.com/mnelsonw210/logo.png",
                                    className='one columns',
                                    style={
                                        'height': '52',
                                        'width': '200',
                                        'float': 'right',
                                        'position': 'relative',
                                    },
                                ),
                            ],
                            className='row'
                        ),
                    ], className='twelve columns', style={'padding-top':'45px', 'padding-bottom':'5px'}
                ),
                html.Div(
                    [
                        html.Div(style={'position':'absolute', 'left':'0', 'right':'0', 'background-color':'black', 'height':'25'}),

                    ], className='twelve columns', style={'padding-bottom':'20px'}
                    # style={'background-image':'url(https://s3-us-west-2.amazonaws.com/mnelsonw210/martin-knize-39793-unsplash2-1-1400x544.jpg)',
                    #                                         'position':'absolute', 'left':'0', 'right':'0', 'height':'150',  'width':'2000',
                    #                                         'background-repeat': 'no-repeat',
                    #                                         'background-position': 'center center',
                    #                                         'overflow': 'hidden',
                    #                                         'object-fit': 'none',
                    #                                         'object-position': 'center'}
                ),
                html.Br(style={'padding-bottom':'20px'}),
            ], className='row', style={'position':'sticky', 'position': '-webkit-sticky', 'top': '0'}
        ),
        html.Br(style={'padding-bottom':'20px'}),
        # html.Div(
        #     [
        #         html.H2(
        #             'Beer Recommendation Assistant',
        #             className='eight columns',
        #         ),
        #         html.Img(
        #             src="https://s3-us-west-2.amazonaws.com/mnelsonw210/logo.png",
        #             className='one columns',
        #             style={
        #                 'height': '52',
        #                 'width': '200',
        #                 'float': 'right',
        #                 'position': 'relative',
        #             },
        #         ),
        #     ],
        #     className='row'
        # ),


        html.Div(
            [
                html.Div(
                    [
                        html.H5(
                        'Based on an Existing Beer',
                        # className='eight columns',
                        ),
                        html.P(
                        'Choose a beer here to see recommended beers for your personal taste!',
                        className='ten columns',
                        ),
                        html.Div(
                            [

                                dcc.RadioItems(
                                    id='beer_list_selector',
                                    options=[
                                        {'label': 'All ', 'value': 'all'},
                                        {'label': 'By Location ', 'value': 'location'},
                                        {'label': 'By Brewery', 'value': 'brewery'},
                                        {'label': 'My Beers ', 'value': 'custom'}
                                    ],
                                    value='custom',
                                    labelStyle={'display': 'inline-block'}
                                ),
                                dcc.Dropdown(
                                    id='beer_indices',
                                    options=beer_index_options,
                                    multi=False,
                                    value=[1]
                                ),
                                html.H1(''),
                                html.Button('Recommend Some Beers!', id='button', style={'background-color':'orange', 'color':'black', 'padding-bottom':'20px'}),
                                html.H1(''),
                            ],
                            className='ten columns',
                            style={'backgroundColor': 'white'},
                        ),
                        html.H1(''),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P('Return beers with a rating no lower than:', style={'font-weight': 'bold'}),
                                        dcc.Slider(
                                            id='rating_slider',
                                            min=0.0,
                                            max=5.0,
                                            value=3.5,
                                            marks=[i for i in range(6)],
                                            step=0.1,
                                            included=False
                                        ),
                                        html.Br(style={'padding-bottom':'20px'}),
                                    ], className='twelve columns',
                                ),
                                html.Div(
                                    [
                                        html.P('Filter Beer List by Brewery:', style={'font-weight': 'bold'}),
                                        dcc.Dropdown(
                                            id='brewery_dropdown',
                                            options=brewery_options,
                                            multi=True,
                                            value=[],
                                        ),
                                        html.Br(style={'padding-bottom':'20px'}),
                                    ], className='twelve columns',
                                ),

                                html.P('Filter Beer List by:', style={'font-weight': 'bold'}),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P('Country', style={'font-weight': 'bold'}),
                                                dcc.Dropdown(
                                                    id='country_dropdown',
                                                    options=country_options,
                                                    multi=True,
                                                    value=[],
                                                ),
                                            ], className='four columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('State/Province', style={'font-weight': 'bold'}),
                                                dcc.Dropdown(
                                                    id='state_dropdown',
                                                    options=state_options,
                                                    multi=True,
                                                    value=[],
                                                ),
                                            ], className='four columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('City', style={'font-weight': 'bold'}),
                                                dcc.Dropdown(
                                                    id='city_dropdown',
                                                    options=city_options,
                                                    multi=True,
                                                    value=[],
                                                ),
                                            ], className='four columns',
                                        ),
                                    ], className='twelve columns',
                                ),
                                dcc.Checklist(
                                    options=[
                                    {'label': 'Also Filter Returned Beers by Location', 'value': 'yes'}
                                    ],
                                    id='location_filter_results',
                                    values=[],
                                ),


                            ],
                            className='ten columns',
                        ),
                    ],
                    className='eight columns',
                    style={'padding-right': '80'}

                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.H6(
                                    '',
                                    id='comparison_beer',
                                    className='twelve columns',
                                    style={'backgroundColor': 'white',
                                    'text-align': 'center',
                                    'margin-top': '0',
                                    'margin-bottom': '0',
                                    'padding-bottom':'20'}
                                ),
                                html.P(
                                    '--- RECOMMENDED BEERS ---',
                                    className='twelve columns',
                                    style={'backgroundColor': 'white',
                                    'text-align': 'center',
                                    'margin-bottom': '0',
                                    'padding-bottom': '20'},
                                ),

                                # individual beer card 1
                                html.Div(
                                    [

                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_name_1',
                                                className='row',
                                                style={'font-weight': 'bold'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_brewery_1',
                                                className='row'
                                            ),
                                        ],
                                        className='nine columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_rating_1',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_abv_1',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_ibu_1',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                        ],
                                        className='three columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    html.P(
                                        '',
                                        id='beer_description_1',
                                        className='twelve columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    ],
                                    className='twelve columns',
                                    style={'backgroundColor': '#D9D9D7',
                                    'border-radius': '10px',
                                    'padding':'5'},
                                ),
                                html.H1(
                                '',
                                className='twelve columns',
                                style={'backgroundColor': 'white'},
                                ),
                                # Beer 2
                                html.Div(
                                    [

                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_name_2',
                                                className='row',
                                                style={'font-weight': 'bold'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_brewery_2',
                                                className='row'
                                            ),
                                        ],
                                        className='nine columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_rating_2',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_abv_2',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_ibu_2',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                        ],
                                        className='three columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    html.P(
                                        '',
                                        id='beer_description_2',
                                        className='twelve columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    ],
                                    className='twelve columns',
                                    style={'backgroundColor': '#D9D9D7',
                                    'border-radius': '10px',
                                    'padding':'5'},
                                ),
                                html.H1(
                                '',
                                className='twelve columns',
                                style={'backgroundColor': 'white'},
                                ),
                                # Beer 3
                                html.Div(
                                    [

                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_name_3',
                                                className='row',
                                                style={'font-weight': 'bold'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_brewery_3',
                                                className='row'
                                            ),
                                        ],
                                        className='nine columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_rating_3',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_abv_3',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_ibu_3',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                        ],
                                        className='three columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    html.P(
                                        '',
                                        id='beer_description_3',
                                        className='twelve columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    ],
                                    className='twelve columns',
                                    style={'backgroundColor': '#D9D9D7',
                                    'border-radius': '10px',
                                    'padding':'5'},
                                ),
                                html.H1(
                                '',
                                className='twelve columns',
                                style={'backgroundColor': 'white'},
                                ),
                                # Beer 4
                                html.Div(
                                    [

                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_name_4',
                                                className='row',
                                                style={'font-weight': 'bold'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_brewery_4',
                                                className='row'
                                            ),
                                        ],
                                        className='nine columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_rating_4',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_abv_4',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_ibu_4',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                        ],
                                        className='three columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    html.P(
                                        '',
                                        id='beer_description_4',
                                        className='twelve columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    ],
                                    className='twelve columns',
                                    style={'backgroundColor': '#D9D9D7',
                                    'border-radius': '10px',
                                    'padding':'5'},
                                ),
                                html.H1(
                                '',
                                className='twelve columns',
                                style={'backgroundColor': 'white'},
                                ),
                                # Beer 5
                                html.Div(
                                    [

                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_name_5',
                                                className='row',
                                                style={'font-weight': 'bold'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_brewery_5',
                                                className='row'
                                            ),
                                        ],
                                        className='nine columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                '',
                                                id='beer_rating_5',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_abv_5',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                            html.P(
                                                '',
                                                id='beer_ibu_5',
                                                className='row',
                                                style={'text-align': 'right'}
                                            ),
                                        ],
                                        className='three columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    html.P(
                                        '',
                                        id='beer_description_5',
                                        className='twelve columns',
                                        style={'backgroundColor': '#D9D9D7'},
                                    ),

                                    ],
                                    className='twelve columns',
                                    style={'backgroundColor': '#D9D9D7',
                                    'border-radius': '10px',
                                    'padding':'5'},
                                ),
                                html.H1(
                                '',
                                className='twelve columns',
                                style={'backgroundColor': 'white'},
                                ),

                            ],
                            className='row',
                            style={'backgroundColor': 'white'},
                        ),

                    ],
                    id='recommended_show',
                    className='four columns',
                    style={'backgroundColor': 'white', 'display':'none'}

                ),
            ],
            className='row'
        ),
    ],
    className='ten columns offset-by-one'
)


## KNN function
def grab_similar_beers(df, index, neighbs = 500):
    new_data_point = np.append(beer_utilities.buildDescVector(google_model, df.loc[index, 'DescriptionTokenized'], vector_dim),[df.loc[index, 'abv'],df.loc[index, 'ibu'],df.loc[index, 'rating'], df.loc[index, 'style_enc']]).reshape([-1,304])
    indices = knn_model.kneighbors(new_data_point, n_neighbors=neighbs+1)[1][0][1:] #have to add a neighbor because it grabs the same beer as the first neighbor

    # used to protect against recommending the same beer
    out = []
    for each in indices:
        if df.loc[each, 'lookup_name'] != df.loc[index, 'lookup_name']:
            out.append(each)

    return beers.loc[out, :]


def generate_name_text(beer_indices, rating_threshold, location_filter_results, city_dropdown, state_dropdown, country_dropdown, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        beers_dff = filter_dataframe(beers_dff, location_filter_results, city_dropdown, state_dropdown, country_dropdown)
        # beers_dff.sort_values(by='rating', ascending=False, inplace=True)
        return beers_dff.iloc[beer_number]['name']
    except:
        return ''

def generate_brewery_text(beer_indices, rating_threshold, location_filter_results, city_dropdown, state_dropdown, country_dropdown, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        beers_dff = filter_dataframe(beers_dff, location_filter_results, city_dropdown, state_dropdown, country_dropdown)
        # beers_dff.sort_values(by='rating', ascending=False, inplace=True)
        return "Brewery: "+beers_dff.iloc[beer_number]['brewery']
    except:
        return ''

def generate_rating_text(beer_indices, rating_threshold, location_filter_results, city_dropdown, state_dropdown, country_dropdown, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        beers_dff = filter_dataframe(beers_dff, location_filter_results, city_dropdown, state_dropdown, country_dropdown)
        # beers_dff.sort_values(by='rating', ascending=False, inplace=True)
        return "Rating: %0.1f" % (beers_dff.iloc[beer_number]['rating'])
    except:
        return ''

def generate_abv_text(beer_indices, rating_threshold, location_filter_results, city_dropdown, state_dropdown, country_dropdown, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        beers_dff = filter_dataframe(beers_dff, location_filter_results, city_dropdown, state_dropdown, country_dropdown)
        # beers_dff.sort_values(by='rating', ascending=False, inplace=True)
        return "ABV: %0.1f" % (beers_dff.iloc[beer_number]['abv']*100)
    except:
        return ''

def generate_ibu_text(beer_indices, rating_threshold, location_filter_results, city_dropdown, state_dropdown, country_dropdown, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        beers_dff = filter_dataframe(beers_dff, location_filter_results, city_dropdown, state_dropdown, country_dropdown)
        # beers_dff.sort_values(by='rating', ascending=False, inplace=True)
        return "IBU: %0.1f" % (beers_dff.iloc[beer_number]['ibu'])
    except:
        return ''

def generate_description_text(beer_indices, rating_threshold, location_filter_results, city_dropdown, state_dropdown, country_dropdown, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        beers_dff = filter_dataframe(beers_dff, location_filter_results, city_dropdown, state_dropdown, country_dropdown)
        # beers_dff.sort_values(by='rating', ascending=False, inplace=True)
        return "Description: "+beers_dff.iloc[beer_number]['description']
    except:
        return ''


def filter_dataframe(beers, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    if location_filter_results == ['yes']:
        if city_dropdown == []:
            if state_dropdown == []:
                if country_dropdown == []:
                    dff = beers
                else:
                    dff = beers[(beers['country'].isin(country_dropdown))]
            else:
                if country_dropdown == []:
                    dff = beers[(beers['state'].isin(state_dropdown))]
                else:
                    dff = beers[(beers['state'].isin(state_dropdown)) &
                                (beers['country'].isin(country_dropdown))]
        else:
            if state_dropdown == []:
                if country_dropdown == []:
                    dff = beers[(beers['city'].isin(city_dropdown))]
                else:
                    dff = beers[(beers['city'].isin(city_dropdown)) &
                                (beers['country'].isin(country_dropdown))]
            else:
                if country_dropdown == []:
                    dff = beers[(beers['city'].isin(city_dropdown)) &
                                (beers['state'].isin(state_dropdown))]
                else:
                    dff = beers[(beers['city'].isin(city_dropdown)) &
                                (beers['state'].isin(state_dropdown)) &
                                (beers['country'].isin(country_dropdown))]
        return dff
    else:
        return beers


@app.callback(Output('beer_indices', 'options'),
              [Input('beer_list_selector', 'value'),
              Input('brewery_dropdown', 'value'),
              Input('city_dropdown', 'value'),
              Input('state_dropdown', 'value'),
              Input('country_dropdown', 'value')])
def display_status(selector, brewery_dropdown, city_dropdown, state_dropdown, country_dropdown):

    if selector == 'location':
        if city_dropdown == []:
            if state_dropdown == []:
                if country_dropdown == []:
                    dff = beers
                else:
                    dff = beers[(beers['country'].isin(country_dropdown))]
            else:
                if country_dropdown == []:
                    dff = beers[(beers['state'].isin(state_dropdown))]
                else:
                    dff = beers[(beers['state'].isin(state_dropdown)) &
                                (beers['country'].isin(country_dropdown))]
        else:
            if state_dropdown == []:
                if country_dropdown == []:
                    dff = beers[(beers['city'].isin(city_dropdown))]
                else:
                    dff = beers[(beers['city'].isin(city_dropdown)) &
                                (beers['country'].isin(country_dropdown))]
            else:
                if country_dropdown == []:
                    dff = beers[(beers['city'].isin(city_dropdown)) &
                                (beers['state'].isin(state_dropdown))]
                else:
                    dff = beers[(beers['city'].isin(city_dropdown)) &
                                (beers['state'].isin(state_dropdown)) &
                                (beers['country'].isin(country_dropdown))]

        beer_index_options = [{'label': str(beers.loc[i, 'lookup_name']), 'value': i} for i in dff.index]
        return beer_index_options

    if selector == 'custom':
        return [{'label': str(beers.loc[i, 'lookup_name']), 'value': i}
                                    for i in beers.index if beers.loc[i, 'lookup_name'] in list(user_profile_beer_list['beer'])]
    if selector == 'brewery':
        dff = beers[(beers['brewery'].isin(brewery_dropdown))]
        beer_index_options = [{'label': str(beers.loc[i, 'lookup_name']), 'value': i} for i in dff.index]
        return beer_index_options

    else:
        beer_index_options = [{'label': str(beers.loc[i, 'lookup_name']), 'value': i} for i in beers.index]
        return beer_index_options

# City Options
@app.callback(Output('city_dropdown', 'options'),
              [Input('state_dropdown', 'value'),
               Input('country_dropdown', 'value'),
               Input('brewery_dropdown', 'value')])
def city_options(state_dropdown, country_dropdown, brewery_dropdown):

    if brewery_dropdown == []:
        if state_dropdown == []:
            if country_dropdown == []:
                dff = beers
            else:
                dff = beers[(beers['country'].isin(country_dropdown))]
        else:
            if country_dropdown == []:
                dff = beers[(beers['state'].isin(state_dropdown))]
            else:
                dff = beers[(beers['state'].isin(state_dropdown)) &
                            (beers['country'].isin(country_dropdown))]
    else:
        if state_dropdown == []:
            if country_dropdown == []:
                dff = beers[(beers['brewery'].isin(brewery_dropdown))]
            else:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['country'].isin(country_dropdown))]
        else:
            if country_dropdown == []:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['state'].isin(state_dropdown))]
            else:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['state'].isin(state_dropdown)) &
                            (beers['country'].isin(country_dropdown))]

    city_options = [{'label': str(i), 'value': i}
                  for i in dff['city'].unique()]
    return city_options

# Country Options
@app.callback(Output('country_dropdown', 'options'),
              [Input('state_dropdown', 'value'),
               Input('city_dropdown', 'value'),
               Input('brewery_dropdown', 'value')])
def city_options(state_dropdown, city_dropdown, brewery_dropdown):

    if brewery_dropdown == []:
        if state_dropdown == []:
            if city_dropdown == []:
                dff = beers
            else:
                dff = beers[(beers['city'].isin(city_dropdown))]
        else:
            if city_dropdown == []:
                dff = beers[(beers['state'].isin(state_dropdown))]
            else:
                dff = beers[(beers['state'].isin(state_dropdown)) &
                            (beers['city'].isin(city_dropdown))]
    else:
        if state_dropdown == []:
            if city_dropdown == []:
                dff = beers[(beers['brewery'].isin(brewery_dropdown))]
            else:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['city'].isin(city_dropdown))]
        else:
            if city_dropdown == []:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['state'].isin(state_dropdown))]
            else:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['state'].isin(state_dropdown)) &
                            (beers['city'].isin(city_dropdown))]

    country_options = [{'label': str(i), 'value': i}
                  for i in dff['country'].unique()]
    return country_options

# Country Options
@app.callback(Output('state_dropdown', 'options'),
              [Input('country_dropdown', 'value'),
               Input('city_dropdown', 'value'),
               Input('brewery_dropdown', 'value')])
def city_options(country_dropdown, city_dropdown, brewery_dropdown):

    if brewery_dropdown == []:
        if country_dropdown == []:
            if city_dropdown == []:
                dff = beers
            else:
                dff = beers[(beers['city'].isin(city_dropdown))]
        else:
            if city_dropdown == []:
                dff = beers[(beers['country'].isin(country_dropdown))]
            else:
                dff = beers[(beers['country'].isin(country_dropdown)) &
                            (beers['city'].isin(city_dropdown))]
    else:
        if country_dropdown == []:
            if city_dropdown == []:
                dff = beers[(beers['brewery'].isin(brewery_dropdown))]
            else:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['city'].isin(city_dropdown))]
        else:
            if city_dropdown == []:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['country'].isin(country_dropdown))]
            else:
                dff = beers[(beers['brewery'].isin(brewery_dropdown)) &
                            (beers['country'].isin(country_dropdown)) &
                            (beers['city'].isin(city_dropdown))]

    state_options = [{'label': str(i), 'value': i}
                  for i in dff['state'].unique()]
    return state_options

# Country Options
@app.callback(Output('brewery_dropdown', 'options'),
              [Input('country_dropdown', 'value'),
               Input('city_dropdown', 'value'),
               Input('state_dropdown', 'value')])
def city_options(country_dropdown, city_dropdown, state_dropdown):

    if state_dropdown == []:
        if country_dropdown == []:
            if city_dropdown == []:
                dff = beers
            else:
                dff = beers[(beers['city'].isin(city_dropdown))]
        else:
            if city_dropdown == []:
                dff = beers[(beers['country'].isin(country_dropdown))]
            else:
                dff = beers[(beers['country'].isin(country_dropdown)) &
                            (beers['city'].isin(city_dropdown))]
    else:
        if country_dropdown == []:
            if city_dropdown == []:
                dff = beers[(beers['state'].isin(state_dropdown))]
            else:
                dff = beers[(beers['state'].isin(state_dropdown)) &
                            (beers['city'].isin(city_dropdown))]
        else:
            if city_dropdown == []:
                dff = beers[(beers['state'].isin(state_dropdown)) &
                            (beers['country'].isin(country_dropdown))]
            else:
                dff = beers[(beers['state'].isin(state_dropdown)) &
                            (beers['country'].isin(country_dropdown)) &
                            (beers['city'].isin(city_dropdown))]

    brewery_options = [{'label': str(i), 'value': i}
                  for i in dff['brewery'].unique()]
    return brewery_options


@app.callback(Output('recommended_show', 'style'),
              [Input('button', 'n_clicks')])
def display_status(button):
    if button:
        return {'backgroundColor': 'white', 'display':'block'}
    else:
        return {'backgroundColor': 'white', 'display':'none'}

# Comparison Beer Name
@app.callback(Output('comparison_beer', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def update_beer_detail_text(button, beer_indices):
    return beers.loc[beer_indices, 'lookup_name']

######################### Recommended Beer #1 Details
@app.callback(Output('beer_name_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_name_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 0)

@app.callback(Output('beer_brewery_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_brewery_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 0)
#
@app.callback(Output('beer_rating_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_rating_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 0)

@app.callback(Output('beer_abv_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_abv_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 0)

@app.callback(Output('beer_ibu_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_ibu_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 0)

@app.callback(Output('beer_description_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_description_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 0)


######################### Recommended Beer #2 Details
@app.callback(Output('beer_name_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_name_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 1)

@app.callback(Output('beer_brewery_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_brewery_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 1)

@app.callback(Output('beer_rating_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_rating_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 1)

@app.callback(Output('beer_abv_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_abv_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 1)

@app.callback(Output('beer_ibu_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_ibu_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 1)

@app.callback(Output('beer_description_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_description_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 1)

######################### Recommended Beer #3 Details
@app.callback(Output('beer_name_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_name_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 2)

@app.callback(Output('beer_brewery_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_brewery_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 2)

@app.callback(Output('beer_rating_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_rating_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 2)

@app.callback(Output('beer_abv_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_abv_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 2)

@app.callback(Output('beer_ibu_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_ibu_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 2)

@app.callback(Output('beer_description_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_description_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 2)

######################### Recommended Beer #4 Details
@app.callback(Output('beer_name_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_name_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 3)

@app.callback(Output('beer_brewery_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_brewery_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 3)

@app.callback(Output('beer_rating_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_rating_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 3)

@app.callback(Output('beer_abv_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_abv_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 3)

@app.callback(Output('beer_ibu_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_ibu_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 3)

@app.callback(Output('beer_description_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_description_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 3)

######################### Recommended Beer #5 Details
@app.callback(Output('beer_name_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_name_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 4)

@app.callback(Output('beer_brewery_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_brewery_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 4)

@app.callback(Output('beer_rating_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_rating_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 4)

@app.callback(Output('beer_abv_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_abv_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 4)

@app.callback(Output('beer_ibu_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_ibu_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 4)

@app.callback(Output('beer_description_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value'),
              State('location_filter_results', 'values'),
              State('city_dropdown', 'value'),
              State('state_dropdown', 'value'),
              State('country_dropdown', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown):
    return generate_description_text(beer_indices, rating_slider, location_filter_results, city_dropdown, state_dropdown, country_dropdown, 4)





if __name__ == '__main__':
    app.run_server(port=8069)
