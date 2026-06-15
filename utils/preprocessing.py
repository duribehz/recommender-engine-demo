import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def load_data():
    base_path = Path(__file__).parent.parent / "data"

    ratings_data = pd.read_csv(base_path / "rating.csv")
    movies_data = pd.read_csv(base_path / "movie.csv")

    movie_ratings_data = pd.merge(ratings_data, movies_data, on="movieId")

    movies_data["genres_clean"] = movies_data["genres"].str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_data["genres_clean"])

    return (
        movies_data,
        ratings_data,
        movie_ratings_data,
        tfidf_matrix
    )