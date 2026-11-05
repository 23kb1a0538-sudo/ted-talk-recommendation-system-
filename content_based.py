
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MATRIX_PATH = PROJECT_DIR / "vectorization_output" / "tfidf_matrix.npz"
METADATA_PATH = PROJECT_DIR / "vectorization_output" / "talk_metadata.csv"

# Load TF-IDF matrix and metadata
tfidf_matrix = load_npz(MATRIX_PATH)
talks = pd.read_csv(METADATA_PATH)

def recommend_talks(talk_id, top_n=10):

    if talk_id < 0 or talk_id >= tfidf_matrix.shape[0]:
        raise ValueError("Invalid talk ID")

    # Calculate similarity with selected talk
    scores = cosine_similarity(
        tfidf_matrix[talk_id],
        tfidf_matrix
    ).flatten()

    # Get most similar talks
    similar_indices = scores.argsort()[::-1]

    similar_indices = [
        i for i in similar_indices if i != talk_id
    ][:top_n]

    results = talks.iloc[similar_indices].copy()
    results["similarity_score"] = scores[similar_indices]

    return results

if __name__ == "__main__":

    print("CONTENT-BASED RECOMMENDATION")

    recommendations = recommend_talks(0, 10)

    print(recommendations.to_string(index=False))
