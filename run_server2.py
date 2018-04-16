# from start_server import app
import flask

app = flask.Flask(__name__)
app.secret_key = 'very secret'
app.config['DEBUG'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config.suppress_callback_exceptions = False


import commercial_beer_viz_breweries, commercial_beer_recommender_custom, commercial_beer_recommender_rating_filter, commercial_beer_viz_styles, recipe_viz

@app.route('/')
def home(name=None):
    return flask.render_template('index.html')

@app.route('/details_recommendations.html')
def recommendation_details(name=None):
    return flask.render_template('details_recommendations.html')

@app.route('/details_recipes.html')
def recipe_details(name=None):
    return flask.render_template('details_recipes.html')

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
    return recipe_viz



if __name__ == '__main__':
    app.secret_key = 'very secret'
    app.run(debug=False, host='0.0.0.0')
