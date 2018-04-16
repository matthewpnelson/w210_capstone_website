import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np


# from dash_apps import ingredient_utils as iu
from dash_apps import data_cleanup_viz as dcv
import warnings

## Dunno if I need this
from flask import Flask
# from flask_cors import CORS

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
from dash_apps import beer_utilities

###############################
from run_server2 import app as server
app = dash.Dash(name='recommend_custom', sharing=True, server=server, url_base_pathname='/recommend_custom')
app.config.suppress_callback_exceptions = False
# app = dash.Dash()
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501
# server = app.server
# CORS(server)

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
beers = pd.read_json('dash_apps/beers_data.json')
# beers['abv'] = beers['abv'] *  100
# Load in Pretrained Label Encoder Model
le = joblib.load('dash_apps/le_model.pkl')

# Load in Pretrained KNN Model
knn_model = joblib.load('dash_apps/knn_model.pkl')

# Specify the Word Vector Dimensionality
vector_dim = 300 #matches the google model

############################################

# Create controls

beer_index_options = [{'label': str(beers.loc[i, 'lookup_name']), 'value': i} for i in beers.index]
beer_style_options = [{'label': str(i), 'value': i}
                  for i in beers['style'].unique()]
brewery_options = [{'label': str(i), 'value': i}
                  for i in beers['brewery'].unique()]
# Flavor Keywords
malty = 'Malty, biscuity, breadlike, grainy, rich, deep, roasty, cereal, cookie-like, coffeeish, caramelly, toffee-like, molasses-like, malt complexity, smoky, sweet, autumnal, burnt cream, scalded milk, oatmeal, rustic, layered'
hoppy = 'Piney, citrusy, grapefruity, earthy, musty, spicy, sharp, bright, fresh, herbal, zippy, lemony, newly-mown lawn, aromatic, floral, springlike, brilliant, sprucelike, juniper-like, minty, pungent, elegant, grassy'
yeasty = 'Fresh-baked bread, clovelike, bubblegum, yeasty, Belgiany, aromatic, tropical, subtle, fruity, clean, banana-like (and for some sour or extreme beers) horseblankety, earthy, musty'
carbony = 'Soft, effervescent, spritzy, sparkling, zippy, pinpoint, bubbly, gentle, low carbonation, highly carbonated'
mouthfeely = 'Rich, full, light, slick, creamy, oily, heavy, velvety, sweet, dry, thick, thin'
alcoholy = 'Warm finish, heat, vodka, esters, pungent, strength'
malty = malty.replace(',', '').lower().split()
hoppy = hoppy.replace(',', '').lower().split()
yeasty = yeasty.replace(',', '').lower().split()
carbony = carbony.replace(',', '').lower().split()
mouthfeely = mouthfeely.replace(',', '').lower().split()
alcoholy = alcoholy.replace(',', '').lower().split()


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
        #             'Beer Recommendation System',
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
        #

        html.Div(
            [
                html.Div(
                    [
                        html.H5(
                        "Build Your Perfect Beer Here, We'll Find the Closest Thing To It",
                        # className='eight columns',
                        ),
                        html.P(
                        'Choose your parameters & write your own description to find similar beers!',
                        className='twelve columns',
                        ),

                        html.H1(''),
                        html.Div(
                            [

                                html.P('Select Beer Style:', style={'font-weight':'bold'}),
                                dcc.Dropdown(
                                    id='beer_style_dropdown',
                                    options=beer_style_options,
                                    multi=False,
                                    value=[],
                                ),
                                html.H1(''),
                                html.H1(''),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P('Select Desired Rating:', style={'font-weight':'bold'}),
                                                dcc.Slider(
                                                    id='rating_slider',
                                                    min=0.0,
                                                    max=5.0,
                                                    value=3.5,
                                                    marks=[i for i in range(6)],
                                                    step=0.1,
                                                    included=False
                                                ),
                                                html.H1(''),
                                                html.H1('', style={'padding-bottom':'10'}),
                                                html.P('Select Desired IBU:', style={'font-weight':'bold'}),
                                                dcc.Slider(
                                                    id='ibu_slider',
                                                    min=0,
                                                    max=300,
                                                    value=80,
                                                    marks={i:i for i in range(0, 301,50)},
                                                    step=5,
                                                    included=False
                                                ),
                                                html.H1(''),
                                                html.H1('', style={'padding-bottom':'10'}),
                                                html.P('Select Desired ABV:', style={'font-weight':'bold'}),
                                                dcc.Slider(
                                                    id='abv_slider',
                                                    min=0,
                                                    max=25,
                                                    value=8,
                                                    marks={i:i for i in range(0, 26,5)},
                                                    step=1,
                                                    included=False
                                                ),
                                            ],
                                            className='twelve columns',
                                        ),
                                    ],
                                    className='twelve columns',
                                    style={'padding-bottom': '50'},
                                ),


                                html.H1(''),
                                html.H1(''),
                                html.P('Write a Description of your Perfect Beer:', style={'font-weight':'bold'}),
                                # html.P('(Or just enter a bunch of flavor adjectives)'),
                                html.Div(
                                    [

                                        dcc.RadioItems(
                                            id='description_selector',
                                            options=[
                                                # {'label': 'All ', 'value': 'all'},
                                                {'label': 'Enter Your Own', 'value': 'Custom'},
                                                {'label': 'Grab A Random Beer Description', 'value': 'Random'},
                                                {'label': 'Build from a List of Flavors', 'value': 'Build'}
                                            ],
                                            value='Random',
                                            labelStyle={'display': 'inline-block'}
                                        ),
                                        dcc.Textarea(
                                            id='description_textbox',
                                            placeholder='Enter description here...',
                                            value='This is a TextArea component',
                                            style={'width': '100%',
                                                'height': '200'},
                                        )

                                    ],
                                    className='twelve columns',
                                    style={'backgroundColor': 'white'},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P('Malts', style={'font-weight':'bold'}),
                                                dcc.Checklist(
                                                    id='flavor_checklist_1',
                                                    options=[{'label': str(i), 'value': i} for i in malty],
                                                    values=[],
                                                ),
                                            ],
                                            className='two columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('Hops', style={'font-weight':'bold'}),
                                                dcc.Checklist(
                                                    id='flavor_checklist_2',
                                                    options=[{'label': str(i), 'value': i} for i in hoppy],
                                                    values=[],
                                                ),
                                            ],
                                            className='two columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('Yeasts', style={'font-weight':'bold'}),
                                                dcc.Checklist(
                                                    id='flavor_checklist_3',
                                                    options=[{'label': str(i), 'value': i} for i in yeasty],
                                                    values=[],
                                                ),
                                            ],
                                            className='two columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('Carbonation', style={'font-weight':'bold'}),
                                                dcc.Checklist(
                                                    id='flavor_checklist_4',
                                                    options=[{'label': str(i), 'value': i} for i in carbony],
                                                    values=[],
                                                ),
                                            ],
                                            className='two columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('Mouthfeel', style={'font-weight':'bold'}),
                                                dcc.Checklist(
                                                    id='flavor_checklist_5',
                                                    options=[{'label': str(i), 'value': i} for i in mouthfeely],
                                                    values=[],
                                                ),
                                            ],
                                            className='two columns',
                                        ),
                                        html.Div(
                                            [
                                                html.P('Alcohol', style={'font-weight':'bold'}),
                                                dcc.Checklist(
                                                    id='flavor_checklist_6',
                                                    options=[{'label': str(i), 'value': i} for i in alcoholy],
                                                    values=[],
                                                ),
                                            ],
                                            className='two columns',
                                        ),
                                    ],
                                    id='flavor_options',
                                    className='twelve columns',
                                    style={'display':'block'}, # swing to 'none' depending on description selector
                                ),

                                html.H1(''),
                                html.Button('Recommend Some Beers!', id='button', style={'background-color':'orange', 'color':'black'}),
                                html.H1('', style={'padding-bottom':'40'})
                            ],
                            className='ten columns',
                        ),

                    ],
                    className='eight columns',
                    style={'padding-right': '80'}
                    # style={'margin-right': '40',
                    # 'margin-left': '40'}
                ),

                html.Div(
                    [
                        html.Div(
                            [

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
                            # id='recommended_show',
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

def grab_beers(abv, ibu, rating, style, description, neighbs = 20):
    abv_in = float(abv) / 100
    ibu_in = float(ibu)
    rating_in = float(rating)
    style_in = le.transform([str(style)])[0]
    description_in = beer_utilities.buildDescVector(google_model, beer_utilities.create_tokens(str(description), t), vector_dim)

    new_data_point = np.append(description_in, abv_in)
    new_data_point = np.append(new_data_point, ibu_in)
    new_data_point = np.append(new_data_point, rating_in)
    new_data_point = np.append(new_data_point, style_in).reshape([-1,304])
    indices = knn_model.kneighbors(new_data_point, n_neighbors=neighbs+1)[1][0][1:] #have to add a neighbor because it grabs the same beer as the first neighbor
    random.shuffle(indices)
    return beers.loc[indices, :]


def generate_name_text(abv, ibu, rating, style, description, beer_number):
    try:
        beers_dff = grab_beers(abv, ibu, rating, style, description[0])
        return beers_dff.iloc[beer_number]['name']
    except:
        return ''

def generate_brewery_text(abv, ibu, rating, style, description, beer_number):
    try:
        beers_dff = grab_beers(abv, ibu, rating, style, description[0])
        return "Brewery: "+beers_dff.iloc[beer_number]['brewery']
    except:
        return ''

def generate_rating_text(abv, ibu, rating, style, description, beer_number):
    try:
        beers_dff = grab_beers(abv, ibu, rating, style, description[0])
        return "Rating: %0.1f" % (beers_dff.iloc[beer_number]['rating'])
    except:
        return ''

def generate_abv_text(abv, ibu, rating, style, description, beer_number):
    try:
        beers_dff = grab_beers(abv, ibu, rating, style, description[0])
        return "ABV: %0.1f" % (beers_dff.iloc[beer_number]['abv']*100)
    except:
        return ''

def generate_ibu_text(abv, ibu, rating, style, description, beer_number):
    try:
        beers_dff = grab_beers(abv, ibu, rating, style, description[0])
        return "IBU: %0.1f" % (beers_dff.iloc[beer_number]['ibu'])
    except:
        return ''

def generate_description_text(abv, ibu, rating, style, description, beer_number):
    try:
        beers_dff = grab_beers(abv, ibu, rating, style, description[0])
        return "Description: "+beers_dff.iloc[beer_number]['description']
    except:
        return ''

@app.callback(Output('description_textbox', 'value'),
              [Input('description_selector', 'value'),
              Input('flavor_checklist_1', 'values'),
              Input('flavor_checklist_2', 'values'),
              Input('flavor_checklist_3', 'values'),
              Input('flavor_checklist_4', 'values'),
              Input('flavor_checklist_5', 'values'),
              Input('flavor_checklist_6', 'values')])
def display_status(selector, flavors1, flavors2, flavors3, flavors4, flavors5, flavors6):
    if selector == 'Custom':
        return "Write a Description..."
    elif selector == 'Random':
        return [x for x in random.sample(set(beers['description'].unique()),1)]
    elif selector == 'Build':
        desc_out = []
        for each in flavors1:
            desc_out.append(each)
        for each in flavors2:
            desc_out.append(each)
        for each in flavors3:
            desc_out.append(each)
        for each in flavors4:
            desc_out.append(each)
        for each in flavors5:
            desc_out.append(each)
        for each in flavors6:
            desc_out.append(each)
        return desc_out
    else:
        return "Write a Description..."

@app.callback(Output('flavor_options', 'style'),
              [Input('description_selector', 'value')])
def display_status(selector):
    if selector == 'Custom' or selector == 'Random':
        return {'display':'none'}
    elif selector == 'Build':
        return {'display':'block'}
    else:
        return {'display':'none'}

@app.callback(Output('recommended_show', 'style'),
              [Input('button', 'n_clicks')])
def display_status(button):
    if button != None:
        return {'backgroundColor': 'white', 'display':'block'}
    else:
        return {'backgroundColor': 'white', 'display':'none'}

@app.callback(Output('abv_slider', 'marks'),
              [Input('beer_style_dropdown', 'value')])
def change_marks(style_dropdown):
    try:
        minimum = beers.loc[beers['style']==style_dropdown, 'abv'].min()*100
        maximum = beers.loc[beers['style']==style_dropdown, 'abv'].max()*100
        if maximum > 25:
            maximum = 26
        marks={
            0: {'label': '0',},
            5: {'label': '5'},
            10: {'label': '10'},
            15: {'label': '15'},
            20: {'label': '20'},
            25: {'label': '25'},
            minimum: {'label': 'min', 'style': {'color': '#e8802e', 'background-color': 'white', 'padding-left': '5', 'padding-right': '5', 'z-index':'1'}},
            maximum: {'label': 'max', 'style': {'color': '#e8802e', 'background-color': 'white', 'padding-left': '5', 'padding-right': '5', 'z-index':'1'}}
        }
        return marks
    except:
        marks={
            0: {'label': '0',},
            5: {'label': '5'},
            10: {'label': '10'},
            15: {'label': '15'},
            20: {'label': '20'},
            25: {'label': '25'}
        }
        return marks

@app.callback(Output('ibu_slider', 'marks'),
              [Input('beer_style_dropdown', 'value')])
def change_marks(style_dropdown):
    try:
        minimum = beers.loc[beers['style']==style_dropdown, 'ibu'].min()
        maximum = beers.loc[beers['style']==style_dropdown, 'ibu'].max()
        if maximum > 300:
            maximum = 310
        marks={
            0: {'label': '0',},
            50: {'label': '50'},
            100: {'label': '100'},
            150: {'label': '150'},
            200: {'label': '200'},
            250: {'label': '250'},
            300: {'label': '300'},

            minimum: {'label': 'min', 'style': {'color': '#e8802e', 'background-color': 'white', 'padding-left': '5', 'padding-right': '5', 'z-index':'1'}},
            maximum: {'label': 'max', 'style': {'color': '#e8802e', 'background-color': 'white', 'padding-left': '5', 'padding-right': '5', 'z-index':'1'}}
        }
        return marks
    except:
        marks={
            0: {'label': '0',},
            50: {'label': '50'},
            100: {'label': '100'},
            150: {'label': '150'},
            200: {'label': '200'},
            250: {'label': '250'},
            300: {'label': '300'}
        }
        return marks

@app.callback(Output('rating_slider', 'marks'),
              [Input('beer_style_dropdown', 'value')])
def change_marks(style_dropdown):
    try:
        minimum = beers.loc[beers['style']==style_dropdown, 'rating'].min()
        maximum = beers.loc[beers['style']==style_dropdown, 'rating'].max()
        if maximum > 5:
            maximum = 5
        marks={
            0: {'label': '0',},
            1: {'label': '1'},
            2: {'label': '2'},
            3: {'label': '3'},
            4: {'label': '4'},
            5: {'label': '5'},
            minimum: {'label': 'min', 'style': {'color': '#e8802e', 'background-color': 'white', 'padding-left': '5', 'padding-right': '5', 'z-index':'1'}},
            maximum: {'label': 'max', 'style': {'color': '#e8802e', 'background-color': 'white', 'padding-left': '5', 'padding-right': '5', 'z-index':'1'}}
        }
        return marks
    except:
        marks={
            0: {'label': '0',},
            1: {'label': '1'},
            2: {'label': '2'},
            3: {'label': '3'},
            4: {'label': '4'},
            5: {'label': '5'}
        }
        return marks


######################### Recommended Beer #1 Details
@app.callback(Output('beer_name_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_name_text(abv, ibu, rating, style, description, 0)

@app.callback(Output('beer_brewery_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_brewery_text(abv, ibu, rating, style, description, 0)

@app.callback(Output('beer_rating_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_rating_text(abv, ibu, rating, style, description, 0)

@app.callback(Output('beer_abv_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_abv_text(abv, ibu, rating, style, description, 0)

@app.callback(Output('beer_ibu_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_ibu_text(abv, ibu, rating, style, description, 0)

@app.callback(Output('beer_description_1', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_description_text(abv, ibu, rating, style, description, 0)


######################### Recommended Beer #2 Details
@app.callback(Output('beer_name_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_name_text(abv, ibu, rating, style, description, 1)

@app.callback(Output('beer_brewery_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_brewery_text(abv, ibu, rating, style, description, 1)

@app.callback(Output('beer_rating_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_rating_text(abv, ibu, rating, style, description, 1)

@app.callback(Output('beer_abv_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_abv_text(abv, ibu, rating, style, description, 1)

@app.callback(Output('beer_ibu_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_ibu_text(abv, ibu, rating, style, description, 1)

@app.callback(Output('beer_description_2', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_description_text(abv, ibu, rating, style, description, 1)

######################### Recommended Beer #3 Details
@app.callback(Output('beer_name_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_name_text(abv, ibu, rating, style, description, 2)

@app.callback(Output('beer_brewery_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_brewery_text(abv, ibu, rating, style, description, 2)

@app.callback(Output('beer_rating_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_rating_text(abv, ibu, rating, style, description, 2)

@app.callback(Output('beer_abv_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_abv_text(abv, ibu, rating, style, description, 2)

@app.callback(Output('beer_ibu_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_ibu_text(abv, ibu, rating, style, description, 2)

@app.callback(Output('beer_description_3', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_description_text(abv, ibu, rating, style, description, 2)

######################### Recommended Beer #4 Details
@app.callback(Output('beer_name_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_name_text(abv, ibu, rating, style, description, 3)

@app.callback(Output('beer_brewery_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_brewery_text(abv, ibu, rating, style, description, 3)

@app.callback(Output('beer_rating_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_rating_text(abv, ibu, rating, style, description, 3)

@app.callback(Output('beer_abv_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_abv_text(abv, ibu, rating, style, description, 3)

@app.callback(Output('beer_ibu_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_ibu_text(abv, ibu, rating, style, description, 3)

@app.callback(Output('beer_description_4', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_description_text(abv, ibu, rating, style, description, 3)

######################### Recommended Beer #5 Details
@app.callback(Output('beer_name_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_name_text(abv, ibu, rating, style, description, 4)

@app.callback(Output('beer_brewery_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_brewery_text(abv, ibu, rating, style, description, 4)

@app.callback(Output('beer_rating_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_rating_text(abv, ibu, rating, style, description, 4)

@app.callback(Output('beer_abv_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_abv_text(abv, ibu, rating, style, description, 4)

@app.callback(Output('beer_ibu_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_ibu_text(abv, ibu, rating, style, description, 4)

@app.callback(Output('beer_description_5', 'children'),
              [Input('button', 'n_clicks')],
              [State('abv_slider', 'value'),
              State('ibu_slider', 'value'),
              State('rating_slider', 'value'),
              State('beer_style_dropdown', 'value'),
              State('description_textbox', 'value')])
def update_beer_detail_text(button, abv, ibu, rating, style, description):
    return generate_description_text(abv, ibu, rating, style, description, 4)




if __name__ == '__main__':
    app.run_server(port=8070)
