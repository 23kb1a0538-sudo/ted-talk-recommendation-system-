
import pandas as pd
import os

file_path = "cleaned_ted_talks.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    print("Cleaned dataset exists!")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

else:
    print("Cleaned file not found!")

