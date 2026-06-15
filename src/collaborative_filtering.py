from surprise import SVD, Dataset, Reader

def get_cf_recommendations(user_id, n, movie_ratings_data, movies_data):
    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(
        movie_ratings_data[["userId", "movieId", "rating"]],
        reader
    )

    algo = SVD()
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    user_movies = movie_ratings_data[movie_ratings_data["userId"] == user_id]
    seen_ids = set(user_movies["movieId"])

    all_movie_ids = movie_ratings_data["movieId"].unique()
    unseen_ids = [mid for mid in all_movie_ids if mid not in seen_ids]

    predictions = [
        (mid, algo.predict(user_id, mid).est)
        for mid in unseen_ids
    ]

    top_ids = [
        x[0] for x in sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    ]

    return movies_data[movies_data["movieId"].isin(top_ids)][["title", "genres"]]