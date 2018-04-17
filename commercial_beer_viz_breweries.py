import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np

import os
# current_folder, current_file = os.path.split(os.path.abspath(__file__))
# current_folder = os.getcwd()
# dash_folder = current_folder+'/dash_apps'
# os.chdir(current_folder)

# from dash_apps import ingredient_utils as iu
# from dash_apps import data_cleanup_viz as dcv
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
# import os
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
# from beer_utilities import

###############################
from run_server2 import app as server
app = dash.Dash(name='explorer_brewery', sharing=True, server=server, url_base_pathname='/explorer_brewery')
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
# google_model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/User/Documents/Berkeley/capstone/GoogleNews-vectors-negative300.bin', binary=True)

# Use NLTK's Tweet Tokenizer
t = TweetTokenizer()

# Load in Pre-cleaned & Adjusted Beers Data
beers = pd.read_json('/root/alegorithm_data/beers_data_py2.json')
beers['abv'] = beers['abv'] *  100
# Load in Pretrained Label Encoder Model
le = joblib.load('/root/alegorithm_data/le_model_py2.pkl')

# Load in Pretrained KNN Model
knn_model = joblib.load('/root/alegorithm_data/knn_model_py2.pkl')


############################################

# Create controls

beer_index_options = [{'label': str(beers.loc[i, 'name'])+': '+str(beers.loc[i, 'rating']), 'value': i} for i in beers.index]
beer_style_options = [{'label': str(i), 'value': i}
                  for i in beers['style'].unique()]
brewery_options = [{'label': str(i), 'value': i}
                  for i in beers['brewery'].unique()]


# Layout
layout = dict(
    autosize=True,
    height=800,
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
                                    'Explore the World of Beers',
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
        #             'Explore the World of Beers',
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
                html.H5(
                'By Brewery',
                className='eight columns',
                ),

                # html.P(
                # 'Closer Dots = Similar Beers (in Beer Style, ABV, IBU, Rating & Description)',
                # className='eight columns',
                # ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.P(
                'Closer Dots = Similar Beers (in Beer Style, ABV, IBU, Rating & Description)',
                className='eight columns',
                ),
                html.Div(
                    [
                        dcc.RadioItems(
                            id='color_selector',
                            options=[
                                {'label': 'Beer Style ', 'value': 'Style'},
                                {'label': 'Brewery', 'value': 'Brewery'},
                            ],
                            value='Brewery',
                            labelStyle={'display': 'inline-block'}
                        ),
                    ],
                    className='two columns',
                    style={'float': 'right',
                           'position': 'relative'},
                ),
                html.Div(
                    [
                        html.P('Plot Colors:'),
                    ],
                    className='one columns',
                    style={'float': 'right',
                           'position': 'relative'},
                ),

            ],
            className='row'
        ),

        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='main_graph')
                    ],
                    className='twelve columns',
                    style={'margin-top': '20'}
                ),
            ],
            className='row'
        ),


        html.Div(
            [
                html.Div(
                    [
                        html.P('Choose a Brewery Here:'),
                        html.Div(
                            [

                                dcc.RadioItems(
                                    id='brewery_selector',
                                    options=[
                                        # {'label': 'All ', 'value': 'all'},
                                        {'label': 'Customize ', 'value': 'custom'},
                                        {'label': 'Random 5', 'value': 'random'},
                                        # {'label': 'Customize ', 'value': 'custom'}
                                    ],
                                    value='random',
                                    labelStyle={'display': 'inline-block'}
                                ),
                                dcc.Dropdown(
                                    id='brewery_dropdown',
                                    options=brewery_options,
                                    multi=True,
                                    value=[],
                                ),
                            ],
                            className='row',
                            style={'backgroundColor': '#D9D9D7'},
                        ),
                    ],
                    className='three columns',
                    style={'margin-right': '40',
                    'margin-left': '40'}
                ),
                html.Div(
                    [
                        html.P('Filter Beer List by Rating (0-5):'),
                        dcc.RangeSlider(
                            id='rating_slider',
                            min=0.0,
                            max=5.0,
                            value=[0,5],
                            marks=[i for i in range(6)],
                            step=0.1
                        ),
                        html.H1(''),
                        html.H1(''),
                        html.P('Filter Beer List by IBU:'),
                        dcc.RangeSlider(
                            id='ibu_slider',
                            min=0,
                            max=500,
                            value=[1,200],
                            marks={i:i for i in range(0, 501,50)},
                            step=5
                        ),
                        html.H1(''),
                        html.H1(''),
                        html.P('Filter Beer List by ABV:'),
                        dcc.RangeSlider(
                            id='abv_slider',
                            min=0,
                            max=25,
                            value=[0,15],
                            marks={i:i for i in range(0, 26,5)},
                            step=1
                        ),
                    ],
                    className='four columns',
                    style={'margin-right': '40',
                    'margin-left': '40'}
                ),
                html.Div(
                    [
                        html.P('Filter Beer List by Beer Style:'),
                         dcc.RadioItems(
                            id='beer_style_selector',
                            options=[
                                {'label': 'All ', 'value': 'all'},
                                {'label': 'Random ', 'value': 'random'},
                                {'label': 'Customize ', 'value': 'custom'}
                            ],
                            value='all',
                            labelStyle={'display': 'inline-block'}
                        ),
                        dcc.Dropdown(
                            id='beer_style_dropdown',
                            options=beer_style_options,
                            multi=True,
                            value=[],
                        ),

                    ],
                    className='three columns',
                    style={'margin-right': '40',
                    'margin-left': '40'}
                ),
                html.Div(
                    [
                        html.H2(''),
                    ],
                    className='six columns'
                ),

            ],
            className='row'
        ),

    ],
    className='ten columns offset-by-one'
)

@app.callback(Output('beer_style_dropdown', 'options'),
              [Input('rating_slider', 'value'),
              Input('ibu_slider', 'value'),
              Input('abv_slider', 'value'),
              Input('brewery_dropdown', 'value')])
def filter_beer_styles(rating_slider, ibu_slider, abv_slider, brewery_dropdown):

    dff = beers[(beers['rating'] > rating_slider[0])
                        & (beers['rating'] < rating_slider[1])
                        & (beers['ibu'] > ibu_slider[0])
                        & (beers['ibu'] < ibu_slider[1])
                        & (beers['abv'] > abv_slider[0])
                        & (beers['abv'] < abv_slider[1])]
                        # & (beers['brewery'].isin(brewery_dropdown))]

    beer_style_options = [{'label': str(i), 'value': i}
                      for i in dff['style'].unique()]
    return beer_style_options


def filter_dataframe(df, rating_slider, ibu_slider, abv_slider, beer_style_dropdown, brewery_dropdown):

    dff = df[(df['rating'] > rating_slider[0])
                    & (df['rating'] < rating_slider[1])
                    & (df['ibu'] > ibu_slider[0])
                    & (df['ibu'] < ibu_slider[1])
                    & (df['abv'] > abv_slider[0])
                    & (df['abv'] < abv_slider[1])
                    & (df['style'].isin(beer_style_dropdown))
                    & (df['brewery'].isin(brewery_dropdown))]
    return dff



# Can't crossfilter the next two callbacks, breaks...
# WORKING
# Radio -> multi
@app.callback(Output('brewery_dropdown', 'value'),
              [Input('brewery_selector', 'value')])
def display_status(selector):
    if selector == 'all':
        return beers['brewery'].unique()
    if selector == 'random':
        return [x for x in random.sample(set(beers['brewery'].unique()),5)]
    else:
        return []

# WORKING
# Radio -> multi
@app.callback(Output('beer_style_dropdown', 'value'),
              [Input('beer_style_selector', 'value'),
              Input('rating_slider', 'value'),
              Input('ibu_slider', 'value'),
              Input('abv_slider', 'value'),
              Input('brewery_dropdown', 'value')])
def display_status(selector, rating_slider, ibu_slider, abv_slider, brewery_dropdown):
    if selector == 'all':
        dff = beers[(beers['rating'] > rating_slider[0])
                            & (beers['rating'] < rating_slider[1])
                            & (beers['ibu'] > ibu_slider[0])
                            & (beers['ibu'] < ibu_slider[1])
                            & (beers['abv'] > abv_slider[0])
                            & (beers['abv'] < abv_slider[1])
                            & (beers['brewery'].isin(brewery_dropdown))]
        return dff['style'].unique()
    if selector == 'random':
        dff = beers[(beers['rating'] > rating_slider[0])
                            & (beers['rating'] < rating_slider[1])
                            & (beers['ibu'] > ibu_slider[0])
                            & (beers['ibu'] < ibu_slider[1])
                            & (beers['abv'] > abv_slider[0])
                            & (beers['abv'] < abv_slider[1])
                            & (beers['brewery'].isin(brewery_dropdown))]
        output = [x for x in random.sample(set(dff['style'].unique()),20)]
        if output == []:
            return ['IPA - New England']
        else:
            return output
    else:
        return []



# Selectors -> main graph
@app.callback(Output('main_graph', 'figure'),
              [Input('rating_slider', 'value'),
              Input('ibu_slider', 'value'),
              Input('abv_slider', 'value'),
              Input('beer_style_dropdown', 'value'),
              Input('brewery_dropdown', 'value'),
              Input('color_selector', 'value')])
def make_main_figure(rating_slider, ibu_slider, abv_slider, beer_style_dropdown, brewery_dropdown, plot_colors_selector):

    beers_dff = filter_dataframe(beers, rating_slider, ibu_slider, abv_slider, beer_style_dropdown, brewery_dropdown)
    traces = []

    if plot_colors_selector == 'Style':
        for beer_style, beers_dfff in beers_dff.groupby('style'):
            trace = dict(
                type='scatter',
                mode='markers',
                x=beers_dfff['tsne_x'],
                y=beers_dfff['tsne_y'],
                text=beers_dfff['lookup_name'],
                # customdata={'abv': beers_dfff['abv']},
                name=beer_style,
                marker=dict(
                    size= 15,
                    opacity=0.6,
                    color=beer_style,
                    colorscale='Earth'
                ),
                hoverinfo='text+name'
            )
            traces.append(trace)
    elif plot_colors_selector == 'Brewery':
        for brewery, beers_dfff in beers_dff.groupby('brewery'):
            trace = dict(
                type='scatter',
                mode='markers',
                x=beers_dfff['tsne_x'],
                y=beers_dfff['tsne_y'],
                text="Name: "+beers_dfff['name'],
                # customdata={'abv': beers_dfff['abv']},
                name=brewery,
                marker=dict(
                    size= 15,
                    opacity=0.6,
                    color=brewery,
                    colorscale='Earth'
                ),
                hoverinfo='text+name'
            )
            traces.append(trace)

    figure = dict(data=traces, layout=layout)
    return figure


if __name__ == '__main__':
    app.run_server(port=8065)
