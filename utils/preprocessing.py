import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    ratings_data = pd.read_csv("data/ratings.csv")
    movies_data = pd.read_csv("data/movies.csv")

    movie_ratings_data = pd.merge(
        ratings_data,
        movies_data,
        on="movieId"
    )

    # TF-IDF usando plot
    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        movies_data["plot"]
    )

    cosine_sim_matrix = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    return (
        movies_data,
        ratings_data,
        movie_ratings_data,
        tfidf_matrix,
        cosine_sim_matrix
    )