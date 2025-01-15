from flask import Flask
from flask_cors import CORS
import pandas as pd
from .model.recommender import MovieRecommender

def create_app():
    app = Flask(__name__)
    CORS(app)  # Tillader cross-origin requests fra din Next.js frontend
    
    # Load data og initialiser recommender
    df = pd.read_csv('app/data/imdb_top_1000(1).csv')
    app.recommender = MovieRecommender(df)
    
    # Registrer routes
    from .api import routes
    app.register_blueprint(routes.bp)
    
    return app