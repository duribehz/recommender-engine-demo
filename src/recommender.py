import pandas as pd
from src.collaborative_filtering import get_cf_recommendations
from src.content_based import get_cb_recommendations

def get_hybrid_recommendations(user_id, n, movie_ratings_data, movies_data, tfidf_matrix):
    cf_recommendations = get_cf_recommendations(
        user_id, n * 2, movie_ratings_data, movies_data
    )

    cb_recommendations = get_cb_recommendations(
        user_id, n * 2, movie_ratings_data, movies_data, tfidf_matrix
    )

    hybrid = (
        pd.concat([cf_recommendations, cb_recommendations])
        .groupby(["title", "genres"])["title"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(n)
    )

    return hybrid[["title", "genres"]]