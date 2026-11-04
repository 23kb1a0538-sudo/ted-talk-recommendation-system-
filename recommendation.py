
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load TED Talk data
df = pd.read_csv("cleaned_ted_talks.csv")

# Load TF-IDF matrix
tfidf_matrix = load_npz(
    "vectorization_output/tfidf_matrix.npz"
)

def recommend_talks(talk_title, top_n=5):

    # Find selected talk
    matches = df[
        df["title"].str.lower() == talk_title.lower()
    ]

    if matches.empty:
        print("TED Talk not found!")
        return

    index = matches.index[0]

    # Calculate similarity scores
    scores = cosine_similarity(
        tfidf_matrix[index],
        tfidf_matrix
    ).flatten()

    # Get top recommendations
    similar_indices = scores.argsort()[::-1]

    print("\nRecommended TED Talks:")
    count = 0

    for i in similar_indices:
        if i == index:
            continue

        print(
            df.iloc[i]["title"],
            "- Similarity:",
            round(scores[i], 3)
        )

        count += 1
        if count == top_n:
            break


# User input
title = input("Enter a TED Talk title: ")

recommend_talks(title)
