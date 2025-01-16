# app/__init__.py
from flask import Flask
from flask_cors import CORS
import pandas as pd
from .model.recommender import MovieRecommender

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Tillad alle origins i udvikling
    
    # Load data og initialiser recommender
    try:
        df = pd.read_csv('app/data/imdb_top_1000.csv')
        app.recommender = MovieRecommender(df)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise e
    
    # Registrer routes
    from .api import routes
    app.register_blueprint(routes.bp)
    
    return app