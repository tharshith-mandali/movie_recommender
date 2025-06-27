import streamlit as st
from recommender import load_data, load_model, get_top_recommendations

st.title("🎬 Movie Recommendation System")
st.write("Rate a few movies to get smart suggestions!")

# Load data and pretrained model
try:
    movies_df, ratings_df = load_data("movies.csv", "ratings.csv")
    model = load_model("svd_model.pkl")
    movie_titles = dict(zip(movies_df.movieId, movies_df.title))
except Exception as e:
    st.error(f"❌ Failed to load data or model: {e}")
    st.stop()

# User selects movies
movie_options = movies_df['title'].tolist()
selected_movies = st.multiselect("🎞️ Choose up to 5 favorite movies:", movie_options)

if len(selected_movies) > 5:
    st.warning("⚠️ You can select a maximum of 5.")
    selected_movies = selected_movies[:5]

# User ratings
user_ratings = {}
for title in selected_movies:
    rating = st.slider(f"⭐ Rate '{title}':", 0.5, 5.0, 3.0, step=0.5)
    movie_id = int(movies_df[movies_df.title == title]['movieId'].values[0])
    user_ratings[movie_id] = rating

# Recommend button
if st.button("🎯 Get Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie.")
    else:
        try:
            results = get_top_recommendations(model, user_ratings, movie_titles, ratings_df)
            st.subheader("🎁 Your Recommendations:")
            for title, score in results:
                st.markdown(f"- **{title}** (predicted: `{score:.2f}` ⭐)")
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")
