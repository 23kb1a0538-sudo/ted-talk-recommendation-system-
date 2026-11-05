
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

METADATA_PATH = PROJECT_DIR / "vectorization_output" / "talk_metadata.csv"
OUTPUT_PATH = BASE_DIR / "user_ratings.csv"

# Load TED Talk metadata
talks = pd.read_csv(METADATA_PATH)

num_talks = len(talks)

if num_talks < 2:
    raise ValueError("At least two TED Talks are required.")

# Generate sample users and ratings
np.random.seed(42)

num_users = 30
ratings_per_user = min(40, num_talks)

records = []

for user in range(1, num_users + 1):

    selected_talks = np.random.choice(
        num_talks,
        size=ratings_per_user,
        replace=False
    )

    for talk_id in selected_talks:

        rating = np.random.randint(1, 6)

        records.append({
            "user_id": user,
            "talk_id": int(talk_id),
            "rating": int(rating)
        })

ratings_df = pd.DataFrame(records)

ratings_df.to_csv(OUTPUT_PATH, index=False)

print("User data created successfully!")
print("Total users:", ratings_df["user_id"].nunique())
print("Total ratings:", len(ratings_df))
print("Saved file:", OUTPUT_PATH)
print(ratings_df.head())
