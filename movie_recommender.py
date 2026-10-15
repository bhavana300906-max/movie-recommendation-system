"""Movie recommender: content-based (genres) + collaborative filtering (SVD).

Expects MovieLens-style files in data/:
    movies.csv   -> movieId, title, genres   (genres separated by '|')
    ratings.csv  -> userId, movieId, rating
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

DATA_DIR = Path("data")


def load_data(data_dir=DATA_DIR):
    movies_path, ratings_path = data_dir / "movies.csv", data_dir / "ratings.csv"
    for p in (movies_path, ratings_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Download MovieLens (e.g. ml-latest-small) and place "
                     "movies.csv and ratings.csv in the data/ folder.")
    movies = pd.read_csv(movies_path).drop_duplicates("movieId").reset_index(drop=True)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


# ---------- Content-based filtering (genres) ----------
def build_similarity(movies):
    """Cosine similarity between movies based on their genres."""
    genres = movies["genres"].fillna("").replace("(no genres listed)", "")
    # Genres are '|'-separated; keep multi-word genres like 'Sci-Fi' intact.
    vectorizer = CountVectorizer(tokenizer=lambda s: [g for g in s.split("|") if g],
                                 token_pattern=None, lowercase=False)
    count_matrix = vectorizer.fit_transform(genres)
    return cosine_similarity(count_matrix, count_matrix)


def find_title(movies, title):
    """Exact (case-insensitive) match first, then substring match."""
    titles = movies["title"]
    exact = movies[titles.str.lower() == title.lower()]
    if not exact.empty:
        return exact.index[0]
    # MovieLens writes "Dark Knight, The (2008)", so also try without a leading article.
    query = title
    for article in ("The ", "A ", "An "):
        if title.lower().startswith(article.lower()):
            query = title[len(article):]
            break
    partial = movies[titles.str.contains(query, case=False, regex=False)]
    if not partial.empty:
        return partial.index[0]
    return None


def get_recommendations(title, movies, cosine_sim, n=10):
    idx = find_title(movies, title)
    if idx is None:
        raise ValueError(f"Movie '{title}' not found in the dataset.")
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    movie_indices = [i for i, _ in sim_scores if i != idx][:n]
    return movies["title"].iloc[movie_indices]


# ---------- Collaborative filtering (SVD) ----------
def train_svd(ratings, test_size=0.25, seed=42):
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=seed)
    algo = SVD(random_state=seed)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return algo, trainset, predictions


def top_n_for_user(algo, trainset, movies, ratings, user_id, n=10):
    """Highest predicted ratings among movies the user has not rated yet."""
    seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    candidates = [m for m in movies["movieId"] if m not in seen]
    scored = sorted(((m, algo.predict(user_id, m).est) for m in candidates),
                    key=lambda x: x[1], reverse=True)[:n]
    titles = movies.set_index("movieId")["title"]
    return pd.DataFrame({"title": [titles[m] for m, _ in scored],
                         "predicted_rating": [round(s, 2) for _, s in scored]})


def main():
    movies, ratings = load_data()

    print("=== Content-based recommendations ===")
    cosine_sim = build_similarity(movies)
    query = "The Dark Knight"
    try:
        print(f"Because you liked '{query}':")
        print(get_recommendations(query, movies, cosine_sim).to_string(index=False))
    except ValueError as e:
        print(e)

    print("\n=== Collaborative filtering (SVD) ===")
    algo, trainset, predictions = train_svd(ratings)
    print("First 5 predictions:")
    for p in predictions[:5]:
        print(p)
    print("RMSE:", round(accuracy.rmse(predictions, verbose=False), 4))
    print("MAE: ", round(accuracy.mae(predictions, verbose=False), 4))

    user_id = ratings["userId"].iloc[0]
    print(f"\nTop 10 picks for user {user_id}:")
    print(top_n_for_user(algo, trainset, movies, ratings, user_id).to_string(index=False))


if __name__ == "__main__":
    main()
