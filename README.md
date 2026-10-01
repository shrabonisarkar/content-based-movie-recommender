# Content-Based Movie Recommender (Baseline Prototype)

A minimal content-based recommendation system: pick a movie, get the Top-K
most similar movies based on genre, keywords, overview text, cast, and
director — no user ratings or behavior involved (that's a later stage).

## 1. Setup

1. Download the **TMDB 5000 Movie Dataset** from Kaggle. You need two files:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`
2. Place both files inside a `data/` folder next to these scripts:
   ```
   movie_recommender/
     data/
       tmdb_5000_movies.csv
       tmdb_5000_credits.csv
     data_preprocessing.py
     build_model.py
     recommend.py
     app.py
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 2. Build the model (run once)

```bash
python data_preprocessing.py   # merges & cleans data -> movies_processed.pkl
python build_model.py          # builds TF-IDF + similarity matrix -> movies.pkl, similarity.pkl
```

## 3. Run the app

```bash
streamlit run app.py
```

---

## How it works, end to end

**Step 1 — Merge & clean (`data_preprocessing.py`)**
The dataset is split across two CSVs (movies, credits), and several columns
(`genres`, `keywords`, `cast`, `crew`) are stored as *stringified JSON*
rather than plain text. We parse those into real lists, pull out the
useful names (genre names, keyword names, top 3 actors, the director),
and merge everything with the movie's `overview` into one combined
`tags` string per movie — this is the "content" the whole system is based on.

A small but important detail: multi-word names like `"Science Fiction"`
or `"Christopher Nolan"` are collapsed into single tokens
(`"sciencefiction"`, `"christophernolan"`). Otherwise TF-IDF would treat
"Science" and "Fiction" as independent, generic words that could
falsely match unrelated movies.

**Step 2 — Vectorize with TF-IDF (`build_model.py`)**
Each movie's `tags` string is converted into a numeric vector using
`TfidfVectorizer`. Intuition:
- Common words across almost every movie (e.g. "life", "story") get a
  **low weight** — they don't help distinguish movies.
- Rare, distinctive words (specific keywords, genre combos, actor/director
  names) get a **high weight** — they're what actually signals similarity.

This produces a `(num_movies × vocabulary_size)` matrix, where each row is
a movie's "fingerprint" in word-space.

**Step 3 — Compare with Cosine Similarity**
We compute the cosine similarity between every pair of movie vectors,
producing a `(num_movies × num_movies)` similarity matrix. Cosine
similarity measures the *angle* between two vectors rather than their raw
size — so a movie with a long, detailed overview doesn't automatically
seem "more similar to everything" just because it has more words.
Score of 1.0 = identical direction (very similar); 0.0 = no shared
distinctive content.

We precompute this whole matrix once and save it to disk (`similarity.pkl`),
so the app doesn't have to redo this expensive computation on every click.

**Step 4 — Recommend (`recommend.py`)**
Given a chosen movie, we look up its row in the similarity matrix (its
similarity score against every other movie), sort those scores in
descending order, drop the movie itself (always similarity 1.0 with
itself), and return the top K.

**Step 5 — UI (`app.py`)**
A thin Streamlit layer: a searchable dropdown of movie titles, a slider
for K, and a button that calls `recommend()` and displays the results.

---

## Known limitations of this baseline (by design — this is a starting point)

- **No personalization** — every user gets the same recommendations for
  the same input movie; the system doesn't know who's asking.
- **No use of ratings/popularity/behavior** — a critically panned movie
  with matching keywords will rank the same as a beloved one. This is
  exactly the gap that **collaborative filtering** or a **hybrid model**
  is meant to fill later.
- **Bag-of-words limitation** — TF-IDF ignores word order and semantic
  meaning ("a story about a robot who loves" vs "a story about loving a
  robot" look identical to it). A future version could use embeddings
  (e.g. sentence-transformers) for semantic similarity instead of raw
  word overlap.
- **Cold start** — a brand-new movie with no/short overview and few
  keywords will get poor-quality tags and therefore poor recommendations.
- **No diversity control** — top-K results can be very "samey" (e.g. all
  five Batman movies). A later version could re-rank for diversity.
- **No explainability shown to the user yet** — we compute a similarity
  score, but don't currently show *which* words/genres/actors drove it.
  That's a natural next feature (e.g. showing overlapping keywords/genres).

These are intentionally left for the next iterations of the project, as
you described (collaborative filtering, hybrid approach, personalization,
explainability, diversity).
