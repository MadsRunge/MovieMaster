import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .utils import preprocess_data
import numpy as np

class MovieRecommender:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.combined_similarity = None
        self.mlb = None
        self.tokenizer = None
        self.label_encoder = None
        self.initialize_system()

    def build_improved_model(self, n_genres, vocab_size, n_directors, n_movies):
        # Model architecture (samme som i din notebook)
        genre_input = Input(shape=(n_genres,), name='genre_input')
        genre_dense = Dense(64, activation='relu')(genre_input)
        genre_dropout = Dropout(0.3)(genre_dense)

        overview_input = Input(shape=(200,), name='overview_input')
        overview_embedding = Embedding(vocab_size, 100)(overview_input)
        overview_lstm = LSTM(128, return_sequences=True)(overview_embedding)
        overview_lstm2 = LSTM(64)(overview_lstm)
        overview_dropout = Dropout(0.3)(overview_lstm2)

        director_input = Input(shape=(n_directors,), name='director_input')
        director_dense = Dense(32, activation='relu')(director_input)
        director_dropout = Dropout(0.3)(director_dense)

        combined = Concatenate()([genre_dropout, overview_dropout, director_dropout])
        dense1 = Dense(256, activation='relu')(combined)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)

        output = Dense(n_movies, activation='softmax')(dropout2)

        model = Model(inputs=[genre_input, overview_input, director_input], outputs=output)
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model

    def initialize_system(self):
        # Preprocess data and create similarity matrices
        genre_matrix, overview_padded, overview_tfidf, director_matrix, \
        self.mlb, self.tokenizer, self.tfidf, self.label_encoder = preprocess_data(self.df)

        # Calculate similarity matrices
        genre_similarity = cosine_similarity(genre_matrix)
        overview_similarity = cosine_similarity(overview_tfidf)
        director_similarity = cosine_similarity(director_matrix)

        # Combine similarities with weights
        self.combined_similarity = 0.15 * genre_similarity + 0.75 * overview_similarity + 0.1 * director_similarity

        # Build model
        self.model = self.build_improved_model(
            genre_matrix.shape[1],
            len(self.tokenizer.word_index) + 1,
            director_matrix.shape[1],
            len(self.df)
        )

    def recommend_movies(self, movie_title, top_k=5):
        """Get recommendations based on a single movie"""
        try:
            # Find movie index
            movie_idx = self.df[self.df['Series_Title'].str.lower() == movie_title.lower()].index[0]

            # Get movie features
            movie_genre = self.mlb.transform([self.df.loc[movie_idx, 'Genre']])
            movie_overview = self.tokenizer.texts_to_sequences([self.df.loc[movie_idx, 'Overview']])
            movie_overview_padded = pad_sequences(movie_overview, maxlen=200, padding='post', truncating='post')

            # Get director features
            director_encoded = self.label_encoder.transform([self.df.loc[movie_idx, 'Director']])
            movie_director = np.eye(len(self.label_encoder.classes_))[director_encoded]

            # Get model predictions
            predictions = self.model.predict(
                [movie_genre, movie_overview_padded, movie_director],
                verbose=0
            )[0]

            # Combine with similarity scores
            similarity_scores = self.combined_similarity[movie_idx]
            combined_scores = 0.99 * predictions + 0.01 * similarity_scores

            # Remove input movie
            combined_scores[movie_idx] = -1

            # Get top recommendations
            top_indices = combined_scores.argsort()[-top_k:][::-1]
            recommendations = self.df.iloc[top_indices][['Series_Title', 'Genre', 'Director', 'IMDB_Rating', 'Overview', 'Released_Year', 'Runtime', 'Poster_Link']]

            return recommendations.to_dict('records')

        except IndexError:
            return {"error": f"Movie '{movie_title}' not found in database."}
            
    def recommend_movies_multi(self, movie_titles, top_k=5):
        """Get recommendations based on multiple input movies"""
        try:
            recommendations_per_movie = {}
            
            for title in movie_titles:
                try:
                    # Find movie index
                    movie_idx = self.df[self.df['Series_Title'].str.lower() == title.lower()].index[0]

                    # Get movie features
                    movie_genre = self.mlb.transform([self.df.loc[movie_idx, 'Genre']])
                    movie_overview = self.tokenizer.texts_to_sequences([self.df.loc[movie_idx, 'Overview']])
                    movie_overview_padded = pad_sequences(movie_overview, maxlen=200, padding='post', truncating='post')

                    # Get director features
                    director_encoded = self.label_encoder.transform([self.df.loc[movie_idx, 'Director']])
                    movie_director = np.eye(len(self.label_encoder.classes_))[director_encoded]

                    # Get model predictions
                    predictions = self.model.predict(
                        [movie_genre, movie_overview_padded, movie_director],
                        verbose=0
                    )[0]

                    # Combine with similarity scores
                    similarity_scores = self.combined_similarity[movie_idx]
                    combined_scores = 0.99 * predictions + 0.01 * similarity_scores

                    # Remove input movies from recommendations
                    for t in movie_titles:
                        exclude_idx = self.df[self.df['Series_Title'].str.lower() == t.lower()].index
                        if len(exclude_idx) > 0:
                            combined_scores[exclude_idx[0]] = -1

                    # Get top recommendations
                    top_indices = combined_scores.argsort()[-top_k:][::-1]
                    recommendations = self.df.iloc[top_indices][['Series_Title', 'Genre', 'Director', 'IMDB_Rating', 'Overview', 'Released_Year', 'Runtime', 'Poster_Link']].to_dict('records')
                    
                    recommendations_per_movie[title] = recommendations
                    
                except IndexError:
                    recommendations_per_movie[title] = {"error": f"Movie '{title}' not found in database."}
                    
            return recommendations_per_movie

        except Exception as e:
            return {"error": str(e)}