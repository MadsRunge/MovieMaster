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

@bp.route('/movie/<title>', methods=['GET'])
def get_movie_details(title):
    """Get details for a specific movie"""
    movie = current_app.recommender.df[
        current_app.recommender.df['Series_Title'].str.lower() == title.lower()
    ]
    
    if movie.empty:
        return jsonify({"error": "Movie not found"}), 404
        
    return jsonify(movie.iloc[0].to_dict())