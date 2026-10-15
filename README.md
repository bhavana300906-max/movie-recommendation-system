# Movie Recommender System

Two recommendation approaches in one script, built on the MovieLens dataset:

1. **Content-based filtering** - recommends movies with similar genres using `CountVectorizer` and cosine similarity.
2. **Collaborative filtering** - predicts user ratings with the SVD matrix-factorization model from the `surprise` library and reports RMSE/MAE.

## Project Structure

```
movie-recommender/
├── data/                  # put movies.csv and ratings.csv here (not included)
├── movie_recommender.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset

Download MovieLens (e.g. `ml-latest-small` from grouplens.org/datasets/movielens) and copy `movies.csv` and `ratings.csv` into `data/`.

- `movies.csv`: `movieId, title, genres` (genres separated by `|`)
- `ratings.csv`: `userId, movieId, rating`

## Installation

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

`scikit-surprise` may need a C compiler on some systems. If installation fails, use Python 3.10-3.12 or install via conda (`conda install -c conda-forge scikit-surprise`).

## Usage

```bash
python movie_recommender.py
```

Change `query` in `main()` to get recommendations for another movie. MovieLens titles often look like `Dark Knight, The (2008)`; the lookup accepts partial matches, so `"Dark Knight"` works.

## Limitations

Genre-only similarity is coarse (every action movie looks alike). Adding tags, cast or plot text would improve it. The collaborative model cannot rate brand-new users or movies (cold start).

## License

MIT
