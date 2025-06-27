import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

def load_data(movies_path, ratings_path):
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    return movies_df, ratings_df

def train_and_save_model(ratings_df, model_path="svd_model.pkl"):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model

def load_model(model_path="svd_model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def get_top_recommendations(model, user_ratings, movie_titles, ratings_df, top_n=5):
    rated_movie_ids = set(user_ratings.keys())
    all_movie_ids = set(movie_titles.keys())

    # Fake user ID for prediction
    user_id = 9999

    predictions = []
    for movie_id in all_movie_ids - rated_movie_ids:
        pred = model.predict(user_id, movie_id, r_ui=0)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = [(movie_titles[mid], score) for mid, score in predictions[:top_n]]
    return top_movies
