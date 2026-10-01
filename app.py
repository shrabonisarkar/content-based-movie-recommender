"""
app.py
------
Simple Streamlit UI for the content-based movie recommender.
Run with:  streamlit run app.py
"""

import streamlit as st
from recommend import load_artifacts, recommend

st.set_page_config(page_title="Movie Recommender (Content-Based)", layout="centered")

st.title("🎬 Content-Based Movie Recommender")
st.caption(
    "Baseline prototype: TF-IDF over genres/keywords/overview/cast/director "
    "+ cosine similarity. Pick a movie you like, and get similar ones."
)


@st.cache_resource
def get_artifacts():
    # @st.cache_resource ensures we load the (potentially large) similarity
    # matrix from disk only ONCE per app session, not on every user click.
    return load_artifacts()


movies, similarity = get_artifacts()

movie_list = sorted(movies["title"].unique())

selected_movie = st.selectbox(
    "Search / select a movie:",
    movie_list,
    index=None,
    placeholder="Start typing a movie title...",
)

top_k = st.slider("How many recommendations?", min_value=3, max_value=20, value=5)

if st.button("Recommend", type="primary", disabled=(selected_movie is None)):
    with st.spinner("Finding similar movies..."):
        results = recommend(selected_movie, movies, similarity, top_k=top_k)

    st.subheader(f"Because you liked '{selected_movie}':")
    for rank, (title, score) in enumerate(results, start=1):
        st.write(f"**{rank}. {title}**  —  similarity score: `{score}`")

st.divider()
st.caption(
    "Note: similarity is purely content-based (text/genre overlap). "
    "It does not use ratings, popularity, or other users' behavior — "
    "that's planned for a later collaborative-filtering / hybrid version."
)
