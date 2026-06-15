import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_movies(movie_id, n, movies_data, tfidf_matrix):
    movie_index = movies_data[movies_data["movieId"] == movie_id].index[0]

    scores = cosine_similarity(
        tfidf_matrix[movie_index], tfidf_matrix
    ).flatten()

    sorted_indices = scores.argsort()[::-1][1:n+1]
    return movies_data.iloc[sorted_indices][["title", "genres"]]


def get_cb_recommendations(user_id, n, movie_ratings_data, movies_data, tfidf_matrix):
    user_movies = movie_ratings_data[movie_ratings_data["userId"] == user_id]

    recommended = []

    for movie_id in user_movies["movieId"]:
        movie_index = movies_data[movies_data["movieId"] == movie_id].index[0]

        scores = cosine_similarity(
            tfidf_matrix[movie_index], tfidf_matrix
        ).flatten()

        sorted_indices = scores.argsort()[::-1][1:n+1]
        recommended += list(zip(sorted_indices, scores[sorted_indices]))

    recommended_indices = [x[0] for x in recommended]
    recommended_movies_data = movies_data.iloc[recommended_indices]

    top_n = (
        recommended_movies_data
        .groupby(["movieId", "title", "genres"])["movieId"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(n)
    )

    return top_n[["title", "genres"]]