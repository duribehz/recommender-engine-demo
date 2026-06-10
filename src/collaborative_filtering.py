from surprise import SVD
from surprise import Dataset
from surprise import Reader


def get_cf_recommendations(
    user_id,
    n,
    movie_ratings_data,
    movies_data
):
    user_movies = movie_ratings_data[
        movie_ratings_data["userId"] == user_id
    ]

    reader = Reader(
        rating_scale=(0.5, 5)
    )

    data = Dataset.load_from_df(
        user_movies[
            ["userId", "movieId", "rating"]
        ],
        reader
    )

    algo = SVD()

    trainset = data.build_full_trainset()

    algo.fit(trainset)

    movie_ids = list(
        movie_ratings_data[
            "movieId"
        ].unique()
    )

    for movie_id in user_movies["movieId"]:
        if movie_id in movie_ids:
            movie_ids.remove(movie_id)

    predictions = []

    for movie_id in movie_ids:

        prediction = algo.predict(
            user_id,
            movie_id
        )

        predictions.append(
            (
                movie_id,
                prediction.est
            )
        )

    top_n_predictions = sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True
    )[:n]

    top_n_movie_ids = [
        x[0]
        for x in top_n_predictions
    ]

    top_n_movies = movies_data[
        movies_data["movieId"].isin(
            top_n_movie_ids
        )
    ]

    return top_n_movies[
        ["title", "genres"]
    ]