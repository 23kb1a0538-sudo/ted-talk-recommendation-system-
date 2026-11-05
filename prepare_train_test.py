
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "user_ratings.csv"
TRAIN_PATH = BASE_DIR / "train_ratings.csv"
TEST_PATH = BASE_DIR / "test_ratings.csv"

# Load ratings
df = pd.read_csv(INPUT_PATH)

# Split ratings separately for each user
train_parts = []
test_parts = []

for user_id, group in df.groupby("user_id"):

    group = group.sample(frac=1, random_state=42)

    if len(group) < 2:
        train_parts.append(group)
        continue

    split_index = max(1, int(len(group) * 0.8))

    train_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

# Save datasets
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Train-test split completed!")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Training file:", TRAIN_PATH)
print("Testing file:", TEST_PATH)

