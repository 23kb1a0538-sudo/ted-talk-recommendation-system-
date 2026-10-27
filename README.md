# TED Talk Recommendation System

## Project Overview
The TED Talk Recommendation System recommends TED Talks based on the content of a talk selected by the user.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit

## Project Workflow
1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Extraction and Vectorization
4. Similarity Calculation
5. TED Talk Recommendations
6. Streamlit Dashboard

## Project Files
- `data_cleaning.py` – Cleans the dataset.
- `vectorization.py` – Converts text into numerical features.
- `similarity.py` – Finds similar TED Talks.
- `dashboard.py` – Provides the web interface.
- `ted_talks.csv` – Original dataset.
- `cleaned_ted_talks.csv` – Processed dataset.

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn streamlit
```

Run the recommendation system:

```bash
python similarity.py
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

## Project Objective
To help users discover TED Talks related to their interests using content-based recommendations.
