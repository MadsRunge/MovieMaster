from flask import Blueprint, jsonify, request, current_app
import pandas as pd
import numpy as np

bp = Blueprint('api', __name__, url_prefix='/api')

def clean_value(value):
    """Convert numpy/pandas values to JSON serializable types"""
    if pd.isna(value) or value is None:
        return None
    elif isinstance(value, (np.integer, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64)):
        return float(value)
    elif isinstance(value, (np.ndarray, list)):
        return [clean_value(v) for v in value]
    elif isinstance(value, str):
        return str(value)
    return value

def format_movie_data(movie):
    """Format movie data to ensure consistent types and structure"""
    # Handle Genre formatting
    genre = movie['Genre']
    if isinstance(genre, str):
        genre = [g.strip() for g in genre.split(',')]
    elif isinstance(genre, list):
        genre = [str(g).strip() for g in genre if g]
    else:
        genre = []
    
    # Create base dictionary
    formatted = {
        'id': str(movie['Series_Title']).lower().replace(' ', '-'),
        'Series_Title': str(movie['Series_Title']),
        'Poster_Link': str(movie['Poster_Link']),
        'Released_Year': clean_value(movie['Released_Year']),
        'Certificate': clean_value(movie['Certificate']),
        'Runtime': clean_value(movie['Runtime']),
        'Genre': genre,
        'IMDB_Rating': clean_value(movie['IMDB_Rating']),
        'Overview': clean_value(movie['Overview']),
        'Meta_score': clean_value(movie['Meta_score']),
        'Director': clean_value(movie['Director']),
        'Star1': clean_value(movie['Star1']),
        'Star2': clean_value(movie['Star2']),
        'Star3': clean_value(movie['Star3']),
        'Star4': clean_value(movie['Star4']),
        'No_of_Votes': clean_value(movie['No_of_Votes']),
        'Gross': clean_value(movie['Gross'])
    }
    
    # Remove any None values
    return {k: v for k, v in formatted.items() if v is not None}

@bp.route('/movies', methods=['GET'])
def get_movies():
    """Return all movies with complete data"""
    try:
        movies = []
        for _, movie in current_app.recommender.df.iterrows():
            try:
                formatted_movie = format_movie_data(movie)
                movies.append(formatted_movie)
            except Exception as e:
                print(f"Error formatting movie: {movie['Series_Title']}, Error: {str(e)}")
                continue
        return jsonify(movies)
    except Exception as e:
        print(f"Error in get_movies: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route('/recommend', methods=['GET'])
def get_recommendations():
    """Get movie recommendations for a single movie"""
    movie_title = request.args.get('title', '')
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    try:
        recommendations = current_app.recommender.recommend_movies(movie_title)
        # Convert recommendations to full data format
        full_recommendations = []
        for rec in recommendations:
            movie_data = current_app.recommender.df[
                current_app.recommender.df['Series_Title'] == rec['Series_Title']
            ].iloc[0]
            full_recommendations.append(format_movie_data(movie_data))
        return jsonify(full_recommendations)
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/recommend-multi', methods=['POST'])
def get_multi_recommendations():
    """Get recommendations based on multiple movies"""
    data = request.get_json()
    
    if not data or 'titles' not in data:
        return jsonify({"error": "No movie titles provided"}), 400
        
    movie_titles = data['titles']
    
    if not isinstance(movie_titles, list):
        return jsonify({"error": "Movie titles must be provided as a list"}), 400
        
    if not 1 <= len(movie_titles) <= 5:
        return jsonify({"error": "Please provide between 1 and 5 movie titles"}), 400
    
    try:
        recommendations = current_app.recommender.recommend_movies_multi(movie_titles)
        # Convert recommendations to full data format
        full_recommendations = []
        for rec in recommendations:
            movie_data = current_app.recommender.df[
                current_app.recommender.df['Series_Title'] == rec['Series_Title']
            ].iloc[0]
            full_recommendations.append(format_movie_data(movie_data))
        return jsonify(full_recommendations)
    except Exception as e:
        print(f"Error in get_multi_recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500