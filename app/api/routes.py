# app/api/routes.py
from flask import Blueprint, jsonify, request, current_app

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/recommend-multi', methods=['POST'])
def get_multi_recommendations():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        if not data or 'titles' not in data:
            return jsonify({"error": "No movie titles provided"}), 400
            
        movie_titles = data['titles']
        print("Processing titles:", movie_titles)  # Debug print
        
        if not isinstance(movie_titles, list):
            return jsonify({"error": "Movie titles must be provided as a list"}), 400
            
        if not 1 <= len(movie_titles) <= 5:
            return jsonify({"error": "Please provide between 1 and 5 movie titles"}), 400
        
        recommendations = current_app.recommender.recommend_movies_multi(movie_titles)
        print("Generated recommendations:", recommendations)  # Debug print
        return jsonify(recommendations)
        
    except Exception as e:
        print(f"Error in get_multi_recommendations: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

@bp.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Route is working!'})