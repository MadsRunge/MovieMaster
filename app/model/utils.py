import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(df):
    # Clean and preprocess genres
    df['Genre'] = df['Genre'].fillna('').apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])

    # Create genre embeddings
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['Genre'])

    # Process overview text
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['Overview'])
    overview_sequences = tokenizer.texts_to_sequences(df['Overview'])
    overview_padded = pad_sequences(overview_sequences, maxlen=200, padding='post', truncating='post')

    # Process directors
    label_encoder = LabelEncoder()
    director_encoded = label_encoder.fit_transform(df['Director'])
    director_matrix = np.eye(len(label_encoder.classes_))[director_encoded]

    # Create TF-IDF features for overview
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    overview_tfidf = tfidf.fit_transform(df['Overview'])

    return genre_matrix, overview_padded, overview_tfidf, director_matrix, mlb, tokenizer, tfidf, label_encoder