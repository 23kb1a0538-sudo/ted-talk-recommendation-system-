import pandas as pd
import os

# Load cleaned dataset
df = pd.read_csv("cleaned_ted_talks.csv")

print("Original Dataset Shape:", df.shape)

# Select useful features
features = df[
    ["title", "description", "tags",
     "main_speaker", "duration", "views",
     "comments", "transcript"]
].copy()

# Handle missing values
features = features.fillna("")

# Create output folder
os.makedirs("feature_output", exist_ok=True)

# Save feature dataset
features.to_csv(
    "feature_output/ted_features.csv",
    index=False
)

# Save feature names
with open("feature_output/feature_columns.txt", "w") as f:
    f.write("\n".join(features.columns))

print("Feature Engineering Completed!")
print("Features Shape:", features.shape)
print("Files saved in feature_output folder")