# from start_server import app
import flask
from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory
import os
from sqlalchemy.orm import sessionmaker
from database.tabledef import *


engine = create_engine('sqlite:///database/alegorithm.db', echo=True)

app = flask.Flask(__name__)
app.secret_key = 'very secret'
app.config['DEBUG'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config.suppress_callback_exceptions = False


import commercial_beer_viz_breweries, commercial_beer_recommender_custom, commercial_beer_recommender_rating_filter, commercial_beer_viz_styles, recipe_viz

@app.route('/')
@app.route('/<name>')
@app.route('/index.html')
def home(name=None):
    return flask.render_template('index.html')

@app.route('/details_recommendations.html')
def recommendation_details(name=None):
    return flask.render_template('details_recommendations.html')

@app.route('/details_recipes.html')
def recipe_details(name=None):
    return flask.render_template('details_recipes.html')

@app.route('/details_architecture.html')
def infrastructure_details(name=None):
    return flask.render_template('details_architecture.html')

@app.route('/user_login', methods=['POST'])
def do_admin_login():
    
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
	
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User)
    query = s.query(User).filter(User.username.in_([POST_USERNAME]), User.password.in_([POST_PASSWORD]) )
    result = query.first()
    if result:
        session['logged_in'] = True
        session['logged_in_username'] = POST_USERNAME
        session['logged_in_name'] = result.name
        print 'logged in name is {}'.format(session.get('logged_in_name'))
        return flask.render_template('index.html', name=session['logged_in_name'])
    else:
        flash('wrong password!')
        return flask.render_template('login.html')
    	

@app.route("/logout")
def logout():
    session['logged_in'] = False
    session['logged_in_as'] = None
    session['logged_in_name'] = None
    return home()
	
@app.route("/register", methods=['POST'])
def register():
    print 'register'	
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    POST_NAME = str(request.form['name'])
    POST_EMAIL = str(request.form['email'])
	
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User)
    query = s.query(User).filter(User.username.in_([POST_USERNAME]))
    result = query.first()
    if result:
        print 'result is {}'.format(result.username)
        flash('user exists!')
        print '{} already exists'.format(POST_NAME)
        return flask.render_template('signup.html')
    else:
        user = User(POST_USERNAME, POST_PASSWORD, POST_NAME, POST_EMAIL)
        s.add(user)
        s.commit()
        session['logged_in'] = True
        session['logged_in_as'] = POST_USERNAME
        session['logged_in_name'] = POST_NAME
        print '{} has been added to the database'.format(POST_NAME)
        return flask.render_template('index.html', name=session.get('logged_in_name'))
	
@app.route('/assets/<path:path>')
def send_assets(path):
    return flask.send_from_directory('assets', path)

@app.route('/dash_apps/<path:path>')
def send_data(path):
    return flask.send_from_directory('data', path)

@app.route('/explorer_brewery')
def explore1():
    return commercial_beer_viz_breweries

@app.route('/explorer_styles')
def explore2():
    return commercial_beer_viz_styles

@app.route('/recommend_custom')
def recommend1():
    return commercial_beer_recommender_custom

@app.route('/recommend_beer')
def recommend2():
    return commercial_beer_recommender_rating_filter

@app.route('/recipes')
def recipeviz():
    print 'Recipes Viz'
    if not session.get('logged_in'):
        print 'Not logged in'
        return flask.render_template('login.html')
    else:
        print 'Already logged in - displaying recipe viz'
        return recipe_viz

@app.route('/signup.html')
def signup(name=None):
    return flask.render_template('signup.html')

@app.route('/login.html')
def login(name=None):
    return flask.render_template('login.html')

if __name__ == '__main__':
    app.secret_key = 'very secret'
    app.run(debug=False, host='0.0.0.0')
