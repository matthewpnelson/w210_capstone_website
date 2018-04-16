### From Commercial Beer Recommender Workbook ###
import pandas as pd
# import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Standard python helper libraries.
import collections
import itertools
import json
import os
import re
import sys
import time
import math

# Numerical manipulation libraries.
import numpy as np
from scipy import stats
import scipy.optimize

# NLTK is the Natural Language Toolkit, and contains several language datasets
# as well as implementations of many popular NLP algorithms.
# HINT: You should look at what is available here when thinking about your project!
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
from beer_utilities import load_commercial_beer_data, create_tokens, buildDescVector

# Google Word2Vec Encoding Model
google_model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/User/Documents/Berkeley/capstone/GoogleNews-vectors-negative300.bin', binary=True)

# Load in Commercial Beer Data and Cleanup

beers = load_commercial_beer_data()

# Convert Commercial Beer Descriptions into Vector Representation
# Tokenize the descriptions

# Use NLTK's Tweet Tokenizer
t = TweetTokenizer()
beers['DescriptionTokenized'] = beers['description'].apply(create_tokens, args=(t,))
beers.head()
X = beers.DescriptionTokenized
Y = beers.rating

# Specify the Word Vector Dimensionality
vector_dim = 300 #matches the google model

X_desc_vecs = np.concatenate([buildDescVector(vec, vector_dim) for vec in map(lambda x: x, X)])

print(X_desc_vecs.shape)
# Probably a more efficient way of doing this but...
start_time = time.time()
for i, beer in enumerate(X_desc_vecs):
    if i == 0:
        X_features = np.append(X_desc_vecs[i],[beers.loc[i, 'abv'],beers.loc[i, 'ibu'],beers.loc[i, 'rating'], beers.loc[i, 'style_enc']])
    else:
        X_features = np.vstack((X_features, np.append(X_desc_vecs[i],[beers.loc[i, 'abv'],beers.loc[i, 'ibu'],beers.loc[i, 'rating'], beers.loc[i, 'style_enc']])))
        if i % 5000 == 0:
            print(X_features.shape)
            print("Elapsed Time: %0.2f s" %((time.time() - start_time)))
X_features.shape


# Train model
# High Dimensionality of the vectors may make KNN ineffective
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

y = beers.loc[:, 'lookup_name']

knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto',
                             metric='l2', p=2, metric_params=None, n_jobs=1)

knn_model.fit(X_features,y)



# Export KNN Model
joblib.dump(knn_model, 'knn_model.pkl')


from sklearn.manifold import TSNE
matrices_embedded = TSNE(n_components=2).fit_transform(X_features)


plotx, ploty = zip(*matrices_embedded)
beers['tsne_x'] = plotx
beers['tsne_y'] = ploty

# Export Beers DF to json
beers.to_json('beers_data.json')
