
import os
import ast
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

# 1. Load the cleaned dataset
input_file = "cleaned_ted_talks_v2.csv"

if not os.path.exists(input_file):
    input_file = "cleaned_ted_talks.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        "Cleaned dataset not found. Check the filename."
    )

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Find columns using common column names
def find_column(possible_names):
    for name in possible_names:
        for col in df.columns:
            if col.lower().strip() == name:
                return col
    return None

title_col = find_column(["title", "name"])
tags_col = find_column(["tags", "topics", "tag"])
desc_col = find_column(["description", "summary"])
transcript_col = find_column(
    ["transcript", "transcripts", "speech"]
)
speaker_col = find_column(
    ["main_speaker", "speaker", "speaker_name"]
)

print("Title column:", title_col)
print("Tags column:", tags_col)
print("Description column:", desc_col)
print("Transcript column:", transcript_col)

if title_col is None:
    raise ValueError(
        "No title column found. Check the printed column names."
    )

# 3. Convert missing values to empty text
def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Convert list-like tags into readable text
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return " ".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass

    return value

# 4. Combine title, tags, description and transcript
# Titles and tags are repeated to give them extra influence.
df["title_text"] = df[title_col].apply(clean_text)

text_parts = [
    df["title_text"] + " " + df["title_text"]
]

if tags_col:
    tags = df[tags_col].apply(clean_text)
    text_parts.append(tags + " " + tags)

if desc_col:
    text_parts.append(df[desc_col].apply(clean_text))

if transcript_col:
    text_parts.append(
        df[transcript_col].apply(clean_text)
    )

df["combined_text"] = text_parts[0]

for part in text_parts[1:]:
    df["combined_text"] = (
        df["combined_text"] + " " + part
    )

# Remove rows with no usable text
df = df[
    df["combined_text"].str.strip().ne("")
].reset_index(drop=True)

# 5. Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    dtype="float32"
)

tfidf_matrix = vectorizer.fit_transform(
    df["combined_text"]
)

print("\nTF-IDF completed!")
print("Talks processed:", len(df))
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 6. Save the vectorizer and numerical matrix
os.makedirs("vectorization_output", exist_ok=True)

joblib.dump(
    vectorizer,
    "vectorization_output/tfidf_vectorizer.joblib"
)

save_npz(
    "vectorization_output/tfidf_matrix.npz",
    tfidf_matrix
)

# Save the talk details in the same row order as the matrix
df.to_csv(
    "vectorization_output/talk_metadata.csv",
    index=False
)

# 7. Calculate and save talk-to-talk similarity
similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)

save_npz(
    "vectorization_output/similarity_matrix.npz",
    similarity_matrix
)

# 8. Show top words for interpretation
feature_names = vectorizer.get_feature_names_out()

print("\nTop 20 TF-IDF vocabulary terms:")
print(feature_names[:20])

print("\nVectorization completed successfully!")
print("Files saved in vectorization_output folder.")
