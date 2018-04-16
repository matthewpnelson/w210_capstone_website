import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np


from dash_apps import ingredient_utils as iu
from dash_apps import data_cleanup_viz as dcv
import warnings

## Dunno if I need this
from flask import Flask
from flask_cors import CORS

### From Commercial Beer Recommender Workbook ###
# import matplotlib.pylab as plt
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import cm

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
from beer_utilities import create_tokens, buildDescVector

###############################
from run_server import app as server
app = dash.Dash(name='recommend_rating', sharing=True, server=server, url_base_pathname='/recommend_rating')
app.config.suppress_callback_exceptions = True
app = dash.Dash()
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501
# server = app.server
CORS(server)

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'  # noqa: E501
    })


colors = {
    'background': '#111111',
    'text': 'black'
}

###### IMPORT DATA ##############

# Google Word2Vec Encoding Model
google_model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/User/Documents/Berkeley/capstone/GoogleNews-vectors-negative300.bin', binary=True)

# Use NLTK's Tweet Tokenizer
t = TweetTokenizer()

# Load in Pre-cleaned & Adjusted Beers Data
beers = pd.read_json('beers_data.json')
# beers['abv'] = beers['abv'] *  100
# Load in Pretrained Label Encoder Model
le = joblib.load('le_model.pkl')

# Load in Pretrained KNN Model
knn_model = joblib.load('knn_model.pkl')

# Specify the Word Vector Dimensionality
vector_dim = 300 #matches the google model

############################################

# Create controls

beer_index_options = [{'label': str(beers.loc[i, 'lookup_name']), 'value': i} for i in beers.index]
beer_style_options = [{'label': str(i), 'value': i}
                  for i in beers['style'].unique()]
brewery_options = [{'label': str(i), 'value': i}
                  for i in beers['brewery'].unique()]

# Input from a User Profile (test)
test_beers = ['New Glarus Brewing Company Spotted Cow','Nano 108 Brewing Company Cascadian Dark Ale (Black Rye IPA)',
'Nano 108 Brewing Company Raspberry And Rose Rice Lager', 'Nano 108 Brewing Company Whiskey Barrel Aged British Burton Ale',
'Nano 108 Brewing Company Woodford Reserve Bourbon Barrel Aged German Doppelbock']
test_ratings = [3.89, 3.77, 3.89, 4.12, 4.34]
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
                html.H2(
                    'Beer Recommendation System',
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


        html.Div(
            [
                html.Div(
                    [
                        html.H5(
                        'Based on a Specific Beer',
                        # className='eight columns',
                        ),
                        html.P(
                        'Choose a beer here to see recommended beers for your personal taste!',
                        className='eight columns',
                        ),
                        html.Div(
                            [

                                dcc.RadioItems(
                                    id='beer_list_selector',
                                    options=[
                                        {'label': 'All ', 'value': 'all'},
                                        # {'label': 'Customize ', 'value': 'custom'},
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
                                html.Button('Recommend Some Beers!', id='button'),
                            ],
                            className='row',
                            style={'backgroundColor': 'white'},
                        ),
                        html.H1(''),
                        html.Div(
                            [
                                html.P('Filter Beer List by Brewery:'),
                                dcc.Dropdown(
                                    id='brewery_dropdown',
                                    options=brewery_options,
                                    multi=True,
                                    value=[],
                                ),
                                html.P('Return beers with a rating no lower than:'),
                                dcc.Slider(
                                    id='rating_slider',
                                    min=0.0,
                                    max=5.0,
                                    value=3.5,
                                    marks=[i for i in range(6)],
                                    step=0.1,
                                    included=False
                                ),
                            ],
                            className='six columns',
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
def grab_similar_beers(df, index, neighbs = 100):
    new_data_point = np.append(buildDescVector(google_model, df.loc[index, 'DescriptionTokenized'], vector_dim),[df.loc[index, 'abv'],df.loc[index, 'ibu'],df.loc[index, 'rating'], df.loc[index, 'style_enc']]).reshape([-1,304])
    indices = knn_model.kneighbors(new_data_point, n_neighbors=neighbs+1)[1][0][1:] #have to add a neighbor because it grabs the same beer as the first neighbor
    # return beers.iloc[indices]
    return beers.loc[indices, :]

def generate_name_text(beer_indices, rating_threshold, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        return beers_dff.iloc[beer_number]['name']
    except:
        return ''

def generate_brewery_text(beer_indices, rating_threshold, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        return "Brewery: "+beers_dff.iloc[beer_number]['brewery']
    except:
        return ''

def generate_rating_text(beer_indices, rating_threshold, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        return "Rating: %0.1f" % (beers_dff.iloc[beer_number]['rating'])
    except:
        return ''

def generate_abv_text(beer_indices, rating_threshold, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        return "ABV: %0.1f" % (beers_dff.iloc[beer_number]['abv']*100)
    except:
        return ''

def generate_ibu_text(beer_indices, rating_threshold, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        return "IBU: %0.1f" % (beers_dff.iloc[beer_number]['ibu'])
    except:
        return ''

def generate_description_text(beer_indices, rating_threshold, beer_number):
    try:
        beers_dff = grab_similar_beers(beers, beer_indices)
        beers_dff = beers_dff[beers_dff['rating'] > rating_threshold]
        return "Description: "+beers_dff.iloc[beer_number]['description']
    except:
        return ''

@app.callback(Output('beer_indices', 'options'),
              [Input('beer_list_selector', 'value'),
              Input('brewery_dropdown', 'value')])
def display_status(selector, brewery_dropdown):
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

@app.callback(Output('recommended_show', 'style'),
              [Input('button', 'n_clicks')])
def display_status(button):
    if button > 0:
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
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_name_text(beer_indices, rating_slider, 0)

@app.callback(Output('beer_brewery_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_brewery_text(beer_indices, rating_slider, 0)

@app.callback(Output('beer_rating_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_rating_text(beer_indices, rating_slider, 0)

@app.callback(Output('beer_abv_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_abv_text(beer_indices, rating_slider, 0)

@app.callback(Output('beer_ibu_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_ibu_text(beer_indices, rating_slider, 0)

@app.callback(Output('beer_description_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_description_text(beer_indices, rating_slider, 0)


######################### Recommended Beer #2 Details
@app.callback(Output('beer_name_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_name_text(beer_indices, rating_slider, 1)

@app.callback(Output('beer_brewery_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_brewery_text(beer_indices, rating_slider, 1)

@app.callback(Output('beer_rating_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_rating_text(beer_indices, rating_slider, 1)

@app.callback(Output('beer_abv_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_abv_text(beer_indices, rating_slider, 1)

@app.callback(Output('beer_ibu_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_ibu_text(beer_indices, rating_slider, 1)

@app.callback(Output('beer_description_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_description_text(beer_indices, rating_slider, 1)

######################### Recommended Beer #3 Details
@app.callback(Output('beer_name_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_name_text(beer_indices, rating_slider, 2)

@app.callback(Output('beer_brewery_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_brewery_text(beer_indices, rating_slider, 2)

@app.callback(Output('beer_rating_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_rating_text(beer_indices, rating_slider, 2)

@app.callback(Output('beer_abv_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_abv_text(beer_indices, rating_slider, 2)

@app.callback(Output('beer_ibu_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_ibu_text(beer_indices, rating_slider, 2)

@app.callback(Output('beer_description_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_description_text(beer_indices, rating_slider, 2)

######################### Recommended Beer #4 Details
@app.callback(Output('beer_name_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_name_text(beer_indices, rating_slider, 3)

@app.callback(Output('beer_brewery_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_brewery_text(beer_indices, rating_slider, 3)

@app.callback(Output('beer_rating_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_rating_text(beer_indices, rating_slider, 3)

@app.callback(Output('beer_abv_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_abv_text(beer_indices, rating_slider, 3)

@app.callback(Output('beer_ibu_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_ibu_text(beer_indices, rating_slider, 3)

@app.callback(Output('beer_description_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_description_text(beer_indices, rating_slider, 3)

######################### Recommended Beer #5 Details
@app.callback(Output('beer_name_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_name_text(beer_indices, rating_slider, 4)

@app.callback(Output('beer_brewery_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_brewery_text(beer_indices, rating_slider, 4)

@app.callback(Output('beer_rating_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_rating_text(beer_indices, rating_slider, 4)

@app.callback(Output('beer_abv_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_abv_text(beer_indices, rating_slider, 4)

@app.callback(Output('beer_ibu_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_ibu_text(beer_indices, rating_slider, 4)

@app.callback(Output('beer_description_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value'),
              State('rating_slider', 'value')])
def update_beer_detail_text(button, beer_indices, rating_slider):
    return generate_description_text(beer_indices, rating_slider, 4)





if __name__ == '__main__':
    app.run_server(port=8069)
