import streamlit as st
from recommender import (
    load_movie_data,
    build_similarity_model,
    recommend_multiple,
    get_movie_poster,
)

# 🔧 Page setup
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Choose either movies or genres to get recommendations.")

# 🔄 Load data and model
df = load_movie_data()
model, tfidf_matrix = build_similarity_model(df)

# User input choice: Movie or Genre
option = st.radio("Select based on:", ["🎥 Movies", "🎭 Genres"])

selected_movies = []
selected_genres = []

if option == "🎥 Movies":
    selected_movies = st.multiselect("Choose one or more movies", df["title"].values)

if option == "🎭 Genres":
    all_genres = sorted({g for genre_list in df["genres"] for g in genre_list.split("|")})
    selected_genres = st.multiselect("Choose one or more genres", all_genres)

# ▶️ On button click
if st.button("Get Recommendations"):
    if option == "🎥 Movies" and not selected_movies:
        st.warning("Please select at least one movie.")
    elif option == "🎭 Genres" and not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        try:
            if selected_movies:
                recommendations = recommend_multiple(
                    selected_movies, df, model, tfidf_matrix, top_n=10
                )
            else:
                # AND logic for genres
                filtered_df = df[df["genres"].apply(
                    lambda g: all(genre in g for genre in selected_genres)
                )]
                recommendations = filtered_df["title"].head(10).tolist()

            if recommendations:
                st.markdown("### 🎥 Recommended Movies:")

                with_posters = []
                without_posters = []

                for movie in recommendations:
                    poster_url = get_movie_poster(movie)
                    if poster_url:
                        with_posters.append((movie, poster_url))
                    else:
                        without_posters.append(movie)

                # Display posters in columns
                cols = st.columns(2)
                for i, (movie, poster_url) in enumerate(with_posters):
                    with cols[i % 2]:
                        st.image(poster_url, width=150)
                        st.markdown(f"**{movie}**")

                # Text-only recommendations
                if without_posters:
                    st.markdown("### 📄 Other Recommended Movies:")
                    for movie in without_posters:
                        st.write(f"🎥 {movie}")
            else:
                st.info("No recommendations found.")
        except Exception as e:
            st.error(f"Recommendation failed: {str(e)}")
