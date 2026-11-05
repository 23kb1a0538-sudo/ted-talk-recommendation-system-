
import pandas as pd
import numpy as np
import pickle

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = BASE_DIR / "collaborative_model.pkl"
TRAIN_PATH = BASE_DIR / "train_ratings.csv"
MATRIX_PATH = PROJECT_DIR / "vectorization_output" / "tfidf_matrix.npz"
METADATA_PATH = PROJECT_DIR / "vectorization_output" / "talk_metadata.csv"
OUTPUT_PATH = BASE_DIR / "hybrid_recommendations.csv"

# Load saved model
with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)

ratings = pd.read_csv(TRAIN_PATH)
tfidf_matrix = load_npz(MATRIX_PATH)
talks = pd.read_csv(METADATA_PATH)

def normalize(values):

    values = np.asarray(values, dtype=float)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def hybrid_recommend(user_id, top_n=10):

    user_map = model_data["user_map"]
    talk_map = model_data["talk_map"]

    if user_id not in user_map:
        raise ValueError("User ID not found.")

    user_index = user_map[user_id]

    # Collaborative filtering scores
    collab_scores = np.zeros(len(talks))

    for talk_id, talk_index in talk_map.items():
        collab_scores[talk_id] = (
            model_data["predicted_ratings"][user_index][talk_index]
        )

    # Content-based scores from highly rated talks
    user_ratings = ratings[ratings["user_id"] == user_id]

    liked_talks = user_ratings[
        user_ratings["rating"] >= 4
    ]

    content_scores = np.zeros(len(talks))

    if len(liked_talks) > 0:

        for _, row in liked_talks.iterrows():

            talk_id = int(row["talk_id"])
            rating = float(row["rating"])

            similarity = cosine_similarity(
                tfidf_matrix[talk_id],
                tfidf_matrix
            ).flatten()

            content_scores += similarity * rating

        content_scores /= liked_talks["rating"].sum()

    # Combine both recommendation scores
    final_scores = (
        0.5 * normalize(content_scores)
        + 0.5 * normalize(collab_scores)
    )

    # Exclude talks already rated by the user
    rated_talks = set(user_ratings["talk_id"])

    for talk_id in rated_talks:
        final_scores[talk_id] = -1

    top_indices = np.argsort(final_scores)[::-1][:top_n]

    results = talks.iloc[top_indices].copy()

    results["talk_id"] = top_indices
    results["hybrid_score"] = final_scores[top_indices]

    return results


if __name__ == "__main__":

    user_id = 1

    recommendations = hybrid_recommend(user_id, 10)

    recommendations.to_csv(OUTPUT_PATH, index=False)

    print("HYBRID RECOMMENDATION RESULTS")
    print(recommendations.to_string(index=False))
    print("Saved:", OUTPUT_PATH)
