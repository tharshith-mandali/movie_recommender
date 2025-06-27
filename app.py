import streamlit as st
from recommender import load_movie_data, build_similarity_model, recommend_movies

# Load and prepare data
st.title("🎬 Genre-Based Movie Recommender")
st.write("Select a movie to get similar recommendations based on genres.")

try:
    df = load_movie_data("movies.csv")
    similarity_matrix = build_similarity_model(df)
    all_titles = df['title'].tolist()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Movie selection
selected_movie = st.selectbox("🎞️ Choose a movie you like:", all_titles)

# Get recommendations
if st.button("🎯 Recommend Similar Movies"):
    try:
        recommendations = recommend_movies(selected_movie, df, similarity_matrix)
        st.subheader("📽️ You might also enjoy:")
        for movie in recommendations:
            st.markdown(f"- **{movie}**")
    except Exception as e:
        st.error(f"❌ Failed to generate recommendations: {e}")
