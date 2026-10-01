"""
data_preprocessing.py
----------------------
Loads the raw TMDB 5000 dataset (movies + credits), cleans it, and builds a
single "tags" text column per movie that summarizes its content:
    genres + keywords + overview + top cast + director

This "tags" column is what we will later feed into TF-IDF.

Why do we bother building one combined text field instead of using each
column separately?
    - It lets us treat "content similarity" as a single text-similarity
      problem: two movies are similar if their tags text is similar.
    - It's simple to reason about and extend later (e.g. add director,
      add production company, weight certain fields more heavily, etc.)
"""

# import pandas as pd
# import ast
# import os

# DATA_DIR = "data"
# MOVIES_CSV = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
# CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
# OUTPUT_PKL = "movies_processed.pkl"


# def parse_names(json_like_str, key="name", limit=None):
#     """
#     TMDB stores fields like genres/keywords/cast/crew as a STRING that looks
#     like a JSON list of dicts, e.g.:
#         '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'

#     ast.literal_eval safely turns that string back into an actual Python
#     list of dicts (safer than eval()). We then just pull out the 'name'
#     field from each dict.

#     limit: if set, only keep the first `limit` entries (e.g. top 3 cast
#     members instead of the entire cast list).
#     """
#     try:
#         items = ast.literal_eval(json_like_str)
#     except (ValueError, SyntaxError):
#         return []

#     names = [item[key] for item in items if key in item]
#     if limit is not None:
#         names = names[:limit]
#     return names


# def get_director(crew_str):
#     """
#     The 'crew' column contains every crew member (editors, producers,
#     directors, etc.) with a 'job' field. We only want the director,
#     since that's usually the most content-defining single person.
#     """
#     try:
#         crew = ast.literal_eval(crew_str)
#     except (ValueError, SyntaxError):
#         return []

#     for member in crew:
#         if member.get("job") == "Director":
#             return [member["name"]]
#     return []


# def collapse_spaces(tokens):
#     """
#     Turns ["Science Fiction", "Christopher Nolan"] into
#     ["sciencefiction", "christophernolan"].

#     Why? Without this, TF-IDF would treat "Science" and "Fiction" as two
#     independent words. That means a movie tagged "Science Fiction" would
#     look similar to any movie containing the word "Science" OR the word
#     "Fiction" alone (e.g. a random word "fiction" in an overview).
#     By collapsing multi-word names into single tokens, "Science Fiction"
#     becomes one atomic keyword that only matches other "Science Fiction"
#     mentions. Same logic applies to actor/director names so "Tom Hanks"
#     isn't split into a generic "Tom" and a generic "Hanks".
#     """
#     return [t.replace(" ", "").lower() for t in tokens]


# def load_and_process():
#     movies = pd.read_csv(MOVIES_CSV)
#     credits = pd.read_csv(CREDITS_CSV)

#     # The credits file uses 'movie_id' while movies uses 'id' -- rename so
#     # we can merge cleanly, and drop the duplicate 'title' column from credits.
#     credits = credits.rename(columns={"movie_id": "id"})
#     credits = credits.drop(columns=["title"])
#     df = movies.merge(credits, on="id")

#     # Keep only the columns we actually need for content-based filtering.
#     df = df[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]

#     # Drop rows with missing overview/title -- can't build meaningful tags
#     # without them.
#     df = df.dropna(subset=["overview", "title"]).reset_index(drop=True)

#     # --- Parse structured columns ---
#     df["genres"] = df["genres"].apply(lambda x: collapse_spaces(parse_names(x)))
#     df["keywords"] = df["keywords"].apply(lambda x: collapse_spaces(parse_names(x)))
#     df["cast"] = df["cast"].apply(lambda x: collapse_spaces(parse_names(x, limit=3)))
#     df["director"] = df["crew"].apply(lambda x: collapse_spaces(get_director(x)))

#     # Overview is free-form text (a sentence or two), so we just split it
#     # into words. We do NOT collapse spaces here -- it's natural language,
#     # not a list of named entities.
#     df["overview_tokens"] = df["overview"].apply(lambda x: x.lower().split())

#     # --- Build the final combined "tags" text ---
#     # Simple concatenation: order doesn't matter to TF-IDF (it's a
#     # "bag of words" model -- see build_model.py for more on this),
#     # only which words appear and how often.
#     df["tags"] = (
#         df["overview_tokens"]
#         + df["genres"]
#         + df["keywords"]
#         + df["cast"]
#         + df["director"]
#     )
#     df["tags"] = df["tags"].apply(lambda tokens: " ".join(tokens))

#     final_df = df[["id", "title", "tags"]].reset_index(drop=True)
#     return final_df


# if __name__ == "__main__":
#     processed = load_and_process()
#     processed.to_pickle(OUTPUT_PKL)
#     print(f"Processed {len(processed)} movies.")
#     print(processed.head(3))
#     print(f"Saved to {OUTPUT_PKL}")

"""
data_preprocessing.py
----------------------
Loads TMDB 5000 movies + credits, preprocesses the data,
creates the tags column, builds a TF-IDF matrix, calculates
cosine similarity, and saves the artifacts required by
recommend.py.
"""

import pandas as pd
import ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MOVIES_CSV = BASE_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = BASE_DIR / "tmdb_5000_credits.csv"

MOVIES_PKL = BASE_DIR / "movies.pkl"
SIMILARITY_PKL = BASE_DIR / "similarity.pkl"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_names(json_like_str, key="name", limit=None):
    try:
        items = ast.literal_eval(json_like_str)
    except (ValueError, SyntaxError, TypeError):
        return []

    names = [item[key] for item in items if key in item]

    if limit is not None:
        names = names[:limit]

    return names


def get_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
    except (ValueError, SyntaxError, TypeError):
        return []

    for member in crew:
        if member.get("job") == "Director":
            return [member["name"]]

    return []


def collapse_spaces(tokens):
    return [t.replace(" ", "").lower() for t in tokens]


# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================

def load_and_process():

    print("Loading datasets...")

    if not MOVIES_CSV.exists():
        raise FileNotFoundError(
            f"Movies CSV not found:\n{MOVIES_CSV}"
        )

    if not CREDITS_CSV.exists():
        raise FileNotFoundError(
            f"Credits CSV not found:\n{CREDITS_CSV}"
        )

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    print(f"Movies loaded: {len(movies)}")
    print(f"Credits loaded: {len(credits)}")

    # Rename movie_id → id
    credits = credits.rename(columns={"movie_id": "id"})

    # Remove duplicate title column if present
    if "title" in credits.columns:
        credits = credits.drop(columns=["title"])

    # Merge datasets
    df = movies.merge(credits, on="id")

    # Keep required columns
    df = df[
        [
            "id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]
    ]

    # Remove missing values
    df = df.dropna(
        subset=["overview", "title"]
    ).reset_index(drop=True)

    print(f"Movies after cleaning: {len(df)}")

    # ========================================================
    # Parse structured columns
    # ========================================================

    df["genres"] = df["genres"].apply(
        lambda x: collapse_spaces(
            parse_names(x)
        )
    )

    df["keywords"] = df["keywords"].apply(
        lambda x: collapse_spaces(
            parse_names(x)
        )
    )

    df["cast"] = df["cast"].apply(
        lambda x: collapse_spaces(
            parse_names(x, limit=3)
        )
    )

    df["director"] = df["crew"].apply(
        lambda x: collapse_spaces(
            get_director(x)
        )
    )

    # Overview
    df["overview_tokens"] = df["overview"].apply(
        lambda x: x.lower().split()
    )

    # ========================================================
    # Create tags
    # ========================================================

    df["tags"] = (
        df["overview_tokens"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["director"]
    )

    df["tags"] = df["tags"].apply(
        lambda tokens: " ".join(tokens)
    )

    # Final dataframe
    final_df = df[
        ["id", "title", "tags"]
    ].reset_index(drop=True)

    return final_df


# ============================================================
# BUILD TF-IDF + COSINE SIMILARITY
# ============================================================

def build_similarity(movies):

    print("Building TF-IDF matrix...")

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        movies["tags"]
    )

    print(
        f"TF-IDF matrix shape: {tfidf_matrix.shape}"
    )

    print("Calculating cosine similarity...")

    similarity = cosine_similarity(tfidf_matrix)

    print(
        f"Similarity matrix shape: {similarity.shape}"
    )

    return similarity


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Step 1: Process data
    movies = load_and_process()

    # Step 2: Build similarity
    similarity = build_similarity(movies)

    # Step 3: Save movies
    movies.to_pickle(MOVIES_PKL)

    # Step 4: Save similarity
    with open(SIMILARITY_PKL, "wb") as f:
        pickle.dump(similarity, f)

    print("\n====================================")
    print("PREPROCESSING COMPLETED")
    print("====================================")

    print(f"Movies saved to:")
    print(MOVIES_PKL)

    print(f"\nSimilarity saved to:")
    print(SIMILARITY_PKL)

    print(f"\nNumber of movies: {len(movies)}")