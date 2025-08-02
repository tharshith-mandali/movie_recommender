import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
import requests

# 🔹 Load and clean movie data from CSV
def load_movie_data():
    df = pd.read_csv("movies.csv")
    df = df.dropna(subset=["genres"])
    return df

# 🔹 Build TF-IDF + Nearest Neighbors similarity model
def build_similarity_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genres"])
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    return model, tfidf_matrix

# 🔹 Recommend based on multiple selected titles
def recommend_multiple(selected_titles, df, model, tfidf_matrix, top_n=10):
    indices = df[df['title'].isin(selected_titles)].index.tolist()

    if not indices:
        return []

    selected_vectors = tfidf_matrix[indices]
    mean_vector = selected_vectors.mean(axis=0)
    mean_vector = np.asarray(mean_vector)

    cosine_similarities = linear_kernel(mean_vector, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[::-1]

    # Remove selected and duplicates
    seen = set(indices)
    unique_recommendations = []
    for idx in similar_indices:
        if idx not in seen:
            unique_recommendations.append(idx)
            seen.add(idx)
        if len(unique_recommendations) >= top_n:
            break

    return df.iloc[unique_recommendations]["title"].tolist()

# 🔹 Get movie poster using TMDB API
def get_movie_poster(title):
    api_key = "02b240bae03be8bed5c8258b401d6c11"  
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
        response = requests.get(url)
        data = response.json()

        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass

    return None
