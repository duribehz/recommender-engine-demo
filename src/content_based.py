import pandas as pd


def get_similar_movies(
    movie_id,
    n,
    movies_data,
    cosine_sim_matrix
):
    movie_index = movies_data[
        movies_data["movieId"] == movie_id
    ].index[0]

    movie_scores = list(
        enumerate(
            cosine_sim_matrix[movie_index]
        )
    )

    sorted_movies = sorted(
        movie_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_movies = []

    for i in range(1, n + 1):
        top_movies.append(
            movies_data.iloc[
                sorted_movies[i][0]
            ]
        )

    return top_movies


def get_cb_recommendations(
    user_id,
    n,
    movie_ratings_data,
    movies_data,
    cosine_sim_matrix
):
    user_movies = movie_ratings_data[
        movie_ratings_data["userId"] == user_id
    ]

    recommended_movies = []

    for movie_id in user_movies["movieId"]:

        movie_index = movies_data[
            movies_data["movieId"] == movie_id
        ].index[0]

        movie_scores = list(
            enumerate(
                cosine_sim_matrix[
                    movie_index
                ]
            )
        )

        sorted_movies = sorted(
            movie_scores,
            key=lambda x: x[1],
            reverse=True
        )[1:n + 1]

        recommended_movies += sorted_movies

    recommended_movie_ids = [
        x[0]
        for x in recommended_movies
    ]

    recommended_movies_data = movies_data[
        movies_data["movieId"].isin(
            recommended_movie_ids
        )
    ]

    top_n_movies = (
        recommended_movies_data
        .groupby(
            ["movieId", "title", "genres"]
        )["movieId"]
        .count()
        .reset_index(name="count")
        .sort_values(
            ["count"],
            ascending=False
        )
        .head(n)
    )

    return top_n_movies[
        ["title", "genres"]
    ]