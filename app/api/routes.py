from flask import Blueprint, jsonify, request, current_app

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/movies', methods=['GET'])
def get_movies():
    """Return list of all movies"""
    movies = current_app.recommender.df['Series_Title'].tolist()
    return jsonify(movies)

@bp.route('/recommend', methods=['GET'])
def get_recommendations():
    """Get movie recommendations"""
    movie_title = request.args.get('title', '')
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    try:
        recommendations = current_app.recommender.recommend_movies(movie_title)
        return jsonify(recommendations)
    except Exception as e:
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
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/movie/<title>', methods=['GET'])
def get_movie_details(title):
    """Get details for a specific movie"""
    movie = current_app.recommender.df[
        current_app.recommender.df['Series_Title'].str.lower() == title.lower()
    ]
    
    if movie.empty:
        return jsonify({"error": "Movie not found"}), 404
        
    return jsonify(movie.iloc[0].to_dict())