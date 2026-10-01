"""
recommend.py
-------------
The actual "recommendation engine" logic, kept separate from the Streamlit
UI so it can be tested/reused independently (e.g. in a notebook, an API,
or a different frontend later).
"""

import pandas as pd
import pickle

MOVIES_PKL = "movies.pkl"
SIMILARITY_PKL = "similarity.pkl"


def load_artifacts():
    movies = pd.read_pickle(MOVIES_PKL)
    with open(SIMILARITY_PKL, "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity


def recommend(movie_title, movies, similarity, top_k=5):
    """
    movie_title : exact title string, must exist in movies['title']
    movies      : dataframe with columns ['id', 'title'], row order matches
                  the similarity matrix
    similarity  : precomputed (num_movies x num_movies) cosine similarity matrix
    top_k       : how many recommendations to return

    Returns a list of (title, similarity_score) tuples.
    """
    matches = movies[movies["title"] == movie_title]
    if matches.empty:
        raise ValueError(f"Movie '{movie_title}' not found in dataset.")

    movie_index = matches.index[0]

    # similarity[movie_index] is this movie's similarity score against
    # every other movie (including itself, which will always be 1.0 --
    # a vector is perfectly similar to itself).
    scores = list(enumerate(similarity[movie_index]))

    # Sort by score descending. We keep the (index, score) pairs together
    # so we can map back to titles after sorting.
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip index 0 in the sorted list -- that's always the movie itself
    # (score = 1.0). We want the next top_k after that.
    top_matches = scores[1: top_k + 1]

    results = [(movies.iloc[i]["title"], round(score, 4)) for i, score in top_matches]
    return results


if __name__ == "__main__":
    movies, similarity = load_artifacts()
    sample_title = movies.iloc[0]["title"]
    print(f"Recommendations for '{sample_title}':")
    for title, score in recommend(sample_title, movies, similarity, top_k=5):
        print(f"  {title}  (similarity: {score})")
