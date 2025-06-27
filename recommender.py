import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load and clean movie data
def load_movie_data(path="movies.csv"):
    df = pd.read_csv(path)
    df = df[df['genres'] != '(no genres listed)'].copy()
    df['genres'] = df['genres'].str.replace('|', ' ')
    return df

# Build TF-IDF matrix
def build_similarity_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    return similarity

# Recommend movies based on input title
def recommend_movies(title, df, similarity_matrix, top_n=5):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

