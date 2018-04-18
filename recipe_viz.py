import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import json
import numpy as np
import os
import copy
import random

# from dash_apps import ingredient_utils as iu
from dash_apps import data_cleanup_viz as dcv
import warnings

from flask import Flask, send_from_directory, render_template

from run_server2 import app as server
app = dash.Dash(name='recipes', sharing=True, server=server, url_base_pathname='/recipes')
app.config.suppress_callback_exceptions = False
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'  # noqa: E501
    })


###### IMPORT DATA ##############

# Load in Reference lists & Cleanup
def reference_load():
    # Grains and Fermentables
    grain_reference = pd.read_csv('/root/alegorithm_data/reference/grain_reference2.csv')
    grain_reference['Aliases'].fillna('', inplace=True)
    # Hops
    hop_reference = pd.read_csv('/root/alegorithm_data/reference/hop_reference2.csv')
    hop_reference['Aliases'].fillna('', inplace=True)
    # Yeasts
    yeast_reference = pd.read_csv('/root/alegorithm_data/reference/yeast_reference.csv')
    yeast_reference['Aliases'].fillna('', inplace=True)
    yeast_reference['Name'].fillna('', inplace=True)

    grain_reference_list = [x.rstrip().lower() for x in grain_reference['Name']]
    grain_reference_list.append([x.rstrip().lower() for x in grain_reference['Aliases']])
    hop_reference_list = [x.rstrip().lower() for x in hop_reference['Name']]
    hop_reference_list.append([x.rstrip().lower() for x in hop_reference['Aliases']])
    yeast_reference_list = [x.rstrip().lower() for x in yeast_reference['Name']]
    yeast_reference_list.append([x.rstrip().lower() for x in yeast_reference['Aliases']])
    return grain_reference_list, hop_reference_list, yeast_reference_list

grain_reference_list, hop_reference_list, yeast_reference_list = reference_load()

# Load in Raw Data, clean-up df, hold in memory
# recipes = iu.load_all_recipes(json_file='recipes_with_matrices_full_8k.json')
recipes = pd.DataFrame(json.load(open('/root/alegorithm_data/recipes_for_viz.json')))
recipes.drop(['level_0'], axis=1, inplace=True)
recipes.reset_index(inplace=True)

to_drop = ["award_winning",
       "bf_co2_unit",
       "bf_priming_amount",
       "bf_priming_method",
       "brew method",
       # "brewer",
       "carbonation_used",
       "display_batch_size",# Eh... Delete if batch_size definitely exists. This is the same as batch size, but it looks like it's variable unit. Batch size looks to always be in liters.,
       "display_boil_size", # Eh... Delete if boil_size definitely exists. This is the same as boil size, but it looks like it's variable unit. Boil size looks to always be in liters.
       "est_abv", # MN: Distribution looks exactly the same as abv. Suggest we remove est_abv.
       "est_fg", # Duplicate to fg_x and fg_y, except it has the units in there too from 'sugar scale'
       "est_color", # MN: Remove, very similar distribution as 'color'
       "est_og", # Duplicate to og_x and og_y, except it has the units in there too from 'sugar scale'
       "forced_carbonation",
       "mash thickness", # Duplicate
       "priming amount", # Duplicate(MN: Don't see the duplicate? should this be kept?)
       "priming method", # Duplicate (MN: Don't see the duplicate? should this be kept?)
       "priming_sugar_name", # Duplicate (MN: Don't see the duplicate? should this be kept?)
       "short title", # remove from dataset
       # "style_x", # Redundant as style_y has a lot more info
       "sugar scale", # This is probably assumed to always be specific gravity. Can't hurt to make sure though. It could technically be brix.
       "bf_co2_level", ## NOT SURE WHAT TO DO WITH THIS?! It's a duplicate of "carbonation" in 20/20 samples I checked.
       "efficiency_y", # Trash. Duplicate of efficiency_x (20/20 I checked were identical)
       "fg_y", # Trash. Duplicate final gravity (20/20 I checked were identical to fg_x)
       "og_x", # Trash. Duplicate of the beerxml one.
       'notes_x',
       'notes_y',
       'boil_size', #duplicate,
       'beerxml',
       'boil_time', #duplicate
       'creation date',
       'link',
       'pitch_rate', #duplicate
       'rating',
       'url',
       'version',
       'views']

recipes.drop(to_drop, axis=1, inplace=True)
recipes = dcv.clean_brewers_friend(recipes)


############################################

# Create controls
beer_index_options = [{'label': str(i), 'value': i}
                  for i in range(recipes.index.max())]
beer_type_options = [{'label': str(i.encode('ascii',errors='ignore')), 'value': i.encode('ascii',errors='ignore')}
                  for i in recipes.style_x.unique()]
beer_author_options = [{'label': str(i.encode('ascii',errors='ignore')), 'value': i.encode('ascii',errors='ignore')}
                  for i in recipes.author.unique() if i is not None]


# Layout
layout = dict(
    autosize=True,
    height=500,
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
    # legend=dict(font=dict(size=10), orientation='h'),
    title='Beer Recipe Structure',
    xaxis={'type': 'linear', 'title': 'Brew Time (min)'},
    yaxis={'title': 'Ingredients (yeasts: oz, fermentables/hops: kg)'},
    legend={'x': 0.9, 'y': 1}

)


# In[]:
# Create app layout
app.layout = html.Div(
    [
        html.Div(
            [
                # html.Div(
                #     [
                #         html.Img(
                #             src="https://s3-us-west-2.amazonaws.com/mnelsonw210/logo.png",
                #             style={
                #                 'height': '52',
                #                 'width': '200',
                #                 'float': 'left',
                #                 'position': 'relative',
                #             },
                #         ),
                #         html.Br(),
                #     ], className='twelve columns'
                # ),
                # html.Br(style={'padding-bottom':'20px'}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    'Beer Recipe Viewer',
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
        html.Div(
            [
                html.Div(
                    [
                        html.P('Choose a Beer Here (Name: Rating):'),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id='beer_name_selector',
                                    options=[
                                        # {'label': 'All ', 'value': 'all'},
                                        {'label': 'Customize ', 'value': 'Custom'},
                                        {'label': 'Random', 'value': 'Random'},
                                        # {'label': 'Customize ', 'value': 'custom'}
                                    ],
                                    value='Random',
                                    labelStyle={'display': 'inline-block'}
                                ),
                                dcc.Dropdown(
                                    id='beer_indices',
                                    options=beer_index_options,
                                    multi=False,
                                    value=[1]
                                ),
                                dcc.Checklist(
                                    options=[
                                    {'label': 'Show only Alegorithm AI Generated Recipes', 'value': 'Alegorithm AI'}
                                    ],
                                    id='beer_author_dropdown',
                                    # options=beer_author_options,
                                    # multi=True,
                                    values=['Alegorithm AI'],
                                ),
                                html.H1(''),
                                html.Button('Retrieve Recipe', id='button', style={'background-color':'orange', 'color':'black'}),
                            ],
                            className='row',
                        ),
                    ],
                    className='six columns'
                ),
                html.Div(
                    [
                        html.P('Filter Beer List by Rating (0-5):'),
                        dcc.RangeSlider(
                            id='rating_slider',
                            min=0.0,
                            max=5.0,
                            value=[3, 4],
                            marks=[i for i in range(6)],
                            step=0.1
                        ),
                        html.H2(''),
                        html.P(''),
                        html.P(''),
                        html.P('Filter Beer List by IBU:'),
                        dcc.RangeSlider(
                            id='ibu_slider',
                            min=0,
                            max=500,
                            value=[10, 80],
                            marks={i:i for i in range(0, 501,50)},
                            step=5
                        ),
                        html.H2(''),
                        html.P(''),
                        html.P(''),
                        html.P('Filter Beer List by ABV:'),
                        dcc.RangeSlider(
                            id='abv_slider',
                            min=0,
                            max=25,
                            value=[0, 10],
                            marks={i:i for i in range(0, 26,5)},
                            step=1
                        ),
                        html.H2(''),
                        html.P(''),
                        html.P(''),
                        html.P('Filter Beer List by Style:'),
                        dcc.Dropdown(
                            id='beer_type_dropdown',
                            options=beer_type_options,
                            multi=True,
                            value=[],
                        ),

                        # html.Div(
                        #     [
                        #         html.Div(
                        #             [
                        #                 html.P('Filter Beer List by Style:'),
                        #                 dcc.Dropdown(
                        #                     id='beer_type_dropdown',
                        #                     options=beer_type_options,
                        #                     multi=True,
                        #                     value=['American Light Lager'],
                        #                 ),
                        #             ],
                        #             className='six columns',
                        #         ),
                        #         html.Div(
                        #             [
                        #                 # html.P('Show only Alegorithm AI Recipes:'),
                        #                 dcc.Checklist(
                        #                     options=[
                        #                     {'label': 'Show only Alegorithm AI Recipes', 'value': 'Alegorithm AI'}
                        #                     ],
                        #                     id='beer_author_dropdown',
                        #                     # options=beer_author_options,
                        #                     # multi=True,
                        #                     values=['Alegorithm AI'],
                        #                 ),
                        #             ],
                        #             className='six columns',
                        #         ),
                        #     ]
                        # ),
                    ],
                    className='six columns'
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.H2(
                    '',
                    className='twelve columns',
                    style={'text-align': 'center'}
                ),
                html.H2(
                    '',
                    id='beer_name_text',
                    className='twelve columns',
                    style={'text-align': 'center'}
                ),
                html.H4(
                    '',
                    id='beer_author_text',
                    className='twelve columns',
                    style={'text-align': 'center'}
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.H5(
                    '',
                    id='beer_type_text',
                    className='two columns'
                ),
                html.H5(
                    '',
                    id='beer_detail_text',
                    className='eight columns',
                    style={'text-align': 'center'}
                ),
                html.H5(
                    '',
                    id='beer_rating_text',
                    className='two columns',
                    style={'text-align': 'right'}
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='main_graph',
                        figure = dict(data=[], layout= dict(
                            autosize=True,
                            height=500,
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
                            # legend=dict(font=dict(size=10), orientation='h'),
                            title='',
                            xaxis={'visible':False, 'type': 'linear', 'ticks': '', 'showticklabels': False},#, 'title': 'Brew Time (min)'},
                            yaxis={'visible':False, 'type': 'linear', 'ticks': '', 'showticklabels': False},#, 'title': 'Ingredients (yeasts: oz, fermentables/hops: kg)'},
                            legend={'x': 0.9, 'y': 1}

                        )))
                    ],
                    className='nine columns',
                    style={'margin-top': '20'}
                ),
                html.Div(
                    [
                        dcc.Graph(id='temp_graph',
                        figure = dict(data=[], layout= dict(
                            autosize=True,
                            height=500,
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
                            # legend=dict(font=dict(size=10), orientation='h'),
                            title='',
                            xaxis={'visible':False, 'type': 'linear', 'ticks': '', 'showticklabels': False},#, 'title': 'Brew Time (min)'},
                            yaxis={'visible':False, 'type': 'linear', 'ticks': '', 'showticklabels': False},#, 'title': 'Ingredients (yeasts: oz, fermentables/hops: kg)'},
                            legend={'x': 0.9, 'y': 1}

                        )))
                    ],
                    className='three columns',
                    style={'margin-top': '20'}
                ),
            ],
            className='row'
        ),
    ],
    className='ten columns offset-by-one'
)


# In[]:
# Calculate Dataframes for use in this Viz, uses an index
# Helper functions
def fetch_recipe_and_temp(recipes, index, grain_reference_list, hop_reference_list, yeast_reference_list):
    df, temp, results, other_info = dcv.clean_and_export(recipes, index, grain_reference_list, hop_reference_list, yeast_reference_list)

    def round_float_3(row):
        return round(row,3)

    df['amount'] = df['amount'].apply(float).apply(round_float_3)
    df['ingredients'] = df['ingredients'].apply(lambda u: str(u.encode('ascii',errors='ignore')))
    df['time'] = df['variable'].apply(float)
    df['type'] = df['type'].apply(str)

    temp['temperature'] = temp['amount'].apply(float)
    temp['type'] = temp['type'].apply(str)
    temp['time'] = temp['variable'].apply(float)

    return df, temp, results, other_info

def fetch_beer_results(indices):

    df, temp, results, other_info = dcv.clean_and_export(recipes, indices, grain_reference_list, hop_reference_list, yeast_reference_list)

    results = results.transpose()
    other_info = other_info.transpose()
    other_info = other_info.loc[['boil time', 'brews', 'carbonation', 'method', #'miscs',
           'yeast_starter', 'boil_gravity_clean', 'boil_size_clean',
           'no_chill_clean', 'pitch_rate_clean', 'primary_temp_clean',
           'starting_mash_thickness_clean'], :]

    try:
        beer_name = results.loc['name', 0].title()
        beer_type = results.loc['style_x', 0].title()
        abv = results.loc['abv_clean', 0]
        ibu = results.loc['ibu_x', 0]
        color = results.loc['color_clean', 0]
        fg = results.loc['fg_clean', 0]
        og = results.loc['og_clean', 0]
        author = results.loc['author', 0]
        beer_rating = results.loc['untappd_rating', 0]
    except:
        return 'n/a', 'n/a', 0, 0, 0, 0, 0, 'n/a', 0

    return beer_name, beer_type, abv, ibu, color, fg, og, author, beer_rating


@app.callback(Output('beer_indices', 'value'),
              [Input('beer_name_selector', 'value'),
              Input('beer_indices', 'options')])
def display_status(selector, options):
    if selector == 'Custom':
        return []
    elif selector == 'Random':
        return options[random.sample(range(len(options)),1)[0]]['value']
    else:
        return []

# slider -> multi
@app.callback(Output('beer_indices', 'options'),
              [Input('rating_slider', 'value'),
              Input('ibu_slider', 'value'),
              Input('abv_slider', 'value'),
              Input('beer_type_dropdown', 'value'),
              Input('beer_author_dropdown', 'values')])
def filter_indices(rating_slider, ibu_slider, abv_slider, beer_type_dropdown, beer_author_dropdown):
    if beer_author_dropdown == ['Alegorithm AI']:
        if beer_type_dropdown == []:
            recipes2 = recipes[(recipes['untappd_rating'] > rating_slider[0])
                            & (recipes['untappd_rating'] < rating_slider[1])
                            & (recipes['ibu_x'] > ibu_slider[0])
                            & (recipes['ibu_x'] < ibu_slider[1])
                            & (recipes['abv_clean'] > abv_slider[0])
                            & (recipes['abv_clean'] < abv_slider[1])
                            & (recipes['author'].isin(beer_author_dropdown))]
        else:
            recipes2 = recipes[(recipes['untappd_rating'] > rating_slider[0])
                            & (recipes['untappd_rating'] < rating_slider[1])
                            & (recipes['ibu_x'] > ibu_slider[0])
                            & (recipes['ibu_x'] < ibu_slider[1])
                            & (recipes['abv_clean'] > abv_slider[0])
                            & (recipes['abv_clean'] < abv_slider[1])
                            & (recipes['style_x'].isin(beer_type_dropdown))
                            & (recipes['author'].isin(beer_author_dropdown))]

    else:
        if beer_type_dropdown == []:
            recipes2 = recipes[(recipes['untappd_rating'] > rating_slider[0])
                            & (recipes['untappd_rating'] < rating_slider[1])
                            & (recipes['ibu_x'] > ibu_slider[0])
                            & (recipes['ibu_x'] < ibu_slider[1])
                            & (recipes['abv_clean'] > abv_slider[0])
                            & (recipes['abv_clean'] < abv_slider[1])]
        else:
            recipes2 = recipes[(recipes['untappd_rating'] > rating_slider[0])
                            & (recipes['untappd_rating'] < rating_slider[1])
                            & (recipes['ibu_x'] > ibu_slider[0])
                            & (recipes['ibu_x'] < ibu_slider[1])
                            & (recipes['abv_clean'] > abv_slider[0])
                            & (recipes['abv_clean'] < abv_slider[1])
                            & (recipes['style_x'].isin(beer_type_dropdown))]

    beer_index_options = [{'label': str(recipes2.loc[i, 'name'])+': '+str(recipes2.loc[i, 'untappd_rating']), 'value': i} for i in recipes2.index]


    return beer_index_options


# Selectors -> Beer Details Text
@app.callback(Output('beer_detail_text', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def update_beer_detail_text(button, beer_indices):
    beer_name, beer_type, abv, ibu, color, fg, og, author, beer_rating = fetch_beer_results(beer_indices)
    return "ABV: %0.1f | IBU: %0.1f | Color: %0.1f | FG: %0.2f | OG: %0.2f" % (abv, ibu, color, fg, og)

# Selectors -> Beer Name Text
@app.callback(Output('beer_name_text', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def update_beer_name_text(button, beer_indices):
    beer_name, beer_type, abv, ibu, color, fg, og, author, beer_rating = fetch_beer_results(beer_indices)
    return "Name: %s" % (beer_name)

# Selectors -> Beer Author Text
@app.callback(Output('beer_author_text', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def update_beer_author_text(button, beer_indices):
    beer_name, beer_type, abv, ibu, color, fg, og, author, beer_rating = fetch_beer_results(beer_indices)
    return "Author: %s" % (author)

# Selectors -> Beer Type Text
@app.callback(Output('beer_type_text', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def update_beer_name_text(button, beer_indices):
    beer_name, beer_type, abv, ibu, color, fg, og, author, beer_rating = fetch_beer_results(beer_indices)
    return "Style: %s" % (beer_type)

# Selectors -> Beer Rating Text
@app.callback(Output('beer_rating_text', 'children'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def update_beer_rating_text(button, beer_indices):
    beer_name, beer_type, abv, ibu, color, fg, og, author, beer_rating = fetch_beer_results(beer_indices)
    return "Rating: %0.2f" % (beer_rating)


# Selectors -> main graph
@app.callback(Output('main_graph', 'figure'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def make_main_figure(button, beer_indices):

    # dff = filter_dataframe(df, well_statuses, well_types, year_slider)
    df, temp, results, other_info = fetch_recipe_and_temp(recipes, beer_indices, grain_reference_list, hop_reference_list, yeast_reference_list)

    traces = []
    for ing_type, dff in df.groupby('type'):
        trace = dict(
            type='scatter',
            mode='markers',
            x=dff['time'],
            y=dff['amount'],
            text=dff['ingredients'],
            customdata=dff['amount'],
            name=ing_type,
            marker=dict(
                size= 30 * np.exp(dff['amount'] / dff.max(axis=0)['amount']),
                opacity=0.6,
                color=ing_type,
                colorscale='Earth'
            )
        )
        traces.append(trace)

    figure = dict(data=traces, layout=layout)
    return figure

# Selectors -> temp graph
@app.callback(Output('temp_graph', 'figure'),
              [Input('button', 'n_clicks')],
              [State('beer_indices', 'value')])
def make_temp_figure(button_click, beer_indices):

    # dff = filter_dataframe(df, well_statuses, well_types, year_slider)
    df, temp, results, other_info = fetch_recipe_and_temp(recipes, beer_indices, grain_reference_list, hop_reference_list, yeast_reference_list)

    layout_individual = copy.deepcopy(layout)
    layout_individual['title'] = 'Beer Cycle Temperature'
    layout_individual['xaxis'] = {'type': 'linear', 'title': 'Brew Time (min)'}
    layout_individual['yaxis'] = {'title': 'Temperature (deg C)'}
    legend={'x': 0, 'y': 1}

    traces = []
    for ing_type, dff in temp.groupby('type'):
        trace = dict(
            type='scatter',
            mode='lines',
            x=dff['time'],
            y=dff['amount'],
            text=dff['ingredients'],
            customdata=dff['ingredients'],
            name='Temperature',
            marker=dict(
                size="spline",
                smoothing="2",
                color=ing_type
            )
        )
        traces.append(trace)

    figure = dict(data=traces, layout=layout_individual)
    return figure



if __name__ == '__main__':
    app.run_server(port=8067)
