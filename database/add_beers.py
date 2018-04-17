import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tabledef import *
import pandas as pd
import numpy as np
import json


def load_beer_data(filename, file_format='json'):
    if file_format == 'json':
        beer_dict = json.load(open(filename+'.json'))
    elif file_format == 'pickle':
        pass
    else:
        print 'Incorrect file format {}'.format(file_format)
    return beer_dict

def utappd_to_sqlite(index, beer):
    '''index, name, description, brewery, style, rating, raters, monthly_stats, unique_stats, total_stats, ibu, abv, label_url, date_added'''
    beer_row = BeerOld(get_index(index), 
					get_name(beer), 
					get_description(beer), 
					get_brewery(beer), 
					get_style(beer),
					get_rating(beer),
					get_raters(beer),
					get_monthly_stats(beer),
					get_unique_stats(beer),
					get_total_stats(beer),
					get_ibu(beer),
					get_abv(beer),
					get_label_url(beer),
					get_date_added(beer)
					)
    return beer_row

def get_index(index):
    return str(index.lstrip().replace('https://untappd.com//b/',''))

def get_name(beer):
    return str(beer['name'].encode('ascii',errors='ignore'))

def get_description(beer):
    return str(beer['description'].encode('ascii',errors='ignore')).strip('\n').replace('\n','').replace('Show Less','').strip()

def get_brewery(beer):
    return str(beer['brewery'].encode('ascii',errors='ignore'))

def get_style(beer):
    return str(beer['style'].encode('ascii',errors='ignore'))

def get_rating(beer):
    return float(beer['rating'])

def get_raters(beer):
    return int(beer['raters'])

def get_monthly_stats(beer):
    return int(beer['stats']['Monthly'])

def get_unique_stats(beer):
    return int(beer['stats']['Unique'])

def get_total_stats(beer):
    return int(beer['stats']['Total'])

def get_ibu(beer):
    return float(beer['ibu'])

def get_abv(beer):
    return float(beer['abv'])

def get_label_url(beer):
    return str(beer['label_url'])

def get_date_added(beer):
    return pd.to_datetime(beer['date_added'])

	
untappd = pd.DataFrame.from_dict(load_beer_data('mehdi_beer_data'), orient='index')
engine = create_engine('sqlite:///alegorithm.db', echo=True)
 
# create a Session
Session = sessionmaker(bind=engine)
session = Session()

for index in untappd.index:
    beer = utappd_to_sqlite(index,untappd.loc[index])
    session.add(beer)
 
# commit the record the database
session.commit()
 
session.commit()