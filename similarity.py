
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load saved files
matrix = load_npz("vectorization_output/tfidf_matrix.npz")
df = pd.read_csv("vectorization_output/talk_metadata.csv")

# Display available TED Talk titles
print("TED TALK RECOMMENDATION SYSTEM")
print("-" * 35)

search_title = input("Enter a TED Talk title: ").strip()

# Find the selected talk
matches = df["title"].fillna("").str.contains(
    search_title, case=False, regex=False
)

if not matches.any():
    print("Sorry, no matching TED Talk found.")
else:
    talk_index = matches[matches].index[0]

    # Calculate similarity scores
    scores = cosine_similarity(
        matrix[talk_index], matrix
    ).flatten()

    # Rank talks by similarity
    ranked_indices = scores.argsort()[::-1]

    print("\nSelected TED Talk:")
    print(df.iloc[talk_index]["title"])

    print("\nRecommended TED Talks:")

    count = 0
    for idx in ranked_indices:
        if idx == talk_index:
            continue

        print(
            f"{df.iloc[idx]['title']} "
            f"- Similarity: {scores[idx]:.3f}"
        )

        count += 1
        if count == 5:
            break
